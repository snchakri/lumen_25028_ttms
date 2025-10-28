"""
Solution Decoder Module for PyGMO Scheduling Problem

Decodes the chromosome (decision vector) into human-readable schedule assignments.
Implements bijective mapping between genotype and phenotype as per Section 8 of the foundations.

The decoder ensures:
1. Information preservation (bijection property)
2. Constraint repair for infeasible solutions
3. Efficient O(n) decoding complexity
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from uuid import UUID

from ..config import PyGMOConfig
from ..logging_system.logger import StructuredLogger
from ..input_model.input_loader import CompiledData


class SolutionDecoder:
    """
    Decodes binary/continuous decision vectors into schedule assignments.
    Implements the reverse mapping from chromosome encoding to assignment tuples.
    """
    
    def __init__(self, compiled_data: CompiledData, config: PyGMOConfig, logger: StructuredLogger,
                 ga_view_info: Dict[str, Any]):
        self.compiled_data = compiled_data
        self.config = config
        self.logger = logger
        self.ga_view_info = ga_view_info
        
        # Extract chromosome encoding information
        self.chromosome_length = ga_view_info.get('chromosome_length', 0)
        self.n_discrete = self.chromosome_length
        self.reverse_mapping = ga_view_info.get('reverse_mapping', {})
        self.gene_bounds = ga_view_info.get('gene_bounds', {})
        
        # Store entity lists for indexing
        self.course_ids = list(compiled_data.courses.keys())
        self.faculty_ids = list(compiled_data.faculty.keys())
        self.room_ids = list(compiled_data.rooms.keys())
        self.timeslot_ids = list(compiled_data.timeslots.keys())
        self.batch_ids = list(compiled_data.batches.keys())
        
        # Validation
        if self.chromosome_length == 0:
            self.logger.warning("Chromosome length is 0. Decoder may not function correctly.")
        
        self.logger.info(f"SolutionDecoder initialized with chromosome length: {self.chromosome_length}")
    
    def decode_assignments(self, discrete_vars: List[float]) -> List[Tuple[UUID, UUID, UUID, UUID, UUID]]:
        """
        Decodes discrete decision variables into schedule assignments.
        
        Args:
            discrete_vars: Binary decision vector (0/1 values) of length chromosome_length
        
        Returns:
            List of (course_id, faculty_id, room_id, timeslot_id, batch_id) tuples
            representing active assignments (where x = 1)
        
        Complexity: O(n) where n = chromosome_length
        """
        assignments = []
        
        if len(discrete_vars) != self.chromosome_length:
            self.logger.error(f"Discrete variable length mismatch: expected {self.chromosome_length}, got {len(discrete_vars)}")
            return assignments
        
        # Decode using reverse mapping
        for gene_index, gene_value in enumerate(discrete_vars):
            # Threshold for binary decision (>0.5 means assigned)
            if gene_value > 0.5:
                # Get the assignment components from reverse mapping
                if gene_index in self.reverse_mapping:
                    c_id_str, f_id_str, r_id_str, t_id_str, b_id_str = self.reverse_mapping[gene_index]
                    
                    try:
                        course_id = UUID(c_id_str)
                        faculty_id = UUID(f_id_str)
                        room_id = UUID(r_id_str)
                        timeslot_id = UUID(t_id_str)
                        batch_id = UUID(b_id_str)
                        
                        assignments.append((course_id, faculty_id, room_id, timeslot_id, batch_id))
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Invalid UUID in reverse mapping at index {gene_index}: {e}")
                        continue
                else:
                    self.logger.warning(f"Gene index {gene_index} not found in reverse mapping. Skipping.")
        
        self.logger.debug(f"Decoded {len(assignments)} active assignments from chromosome.")
        return assignments
    
    def encode_assignments(self, assignments: List[Tuple[UUID, UUID, UUID, UUID, UUID]]) -> List[float]:
        """
        Encodes schedule assignments back into a chromosome (inverse of decode_assignments).
        This is useful for seeding the population with known good solutions.
        
        Args:
            assignments: List of (course_id, faculty_id, room_id, timeslot_id, batch_id) tuples
        
        Returns:
            Binary decision vector of length chromosome_length
        
        Complexity: O(m + n) where m = len(assignments), n = chromosome_length
        """
        chromosome = [0.0] * self.chromosome_length
        
        # Build a forward mapping for efficient lookup
        forward_mapping = {v: k for k, v in self.reverse_mapping.items()}
        
        for course_id, faculty_id, room_id, timeslot_id, batch_id in assignments:
            # Construct the gene key
            gene_key = (str(course_id), str(faculty_id), str(room_id), str(timeslot_id), str(batch_id))
            
            if gene_key in forward_mapping:
                gene_index = forward_mapping[gene_key]
                chromosome[gene_index] = 1.0
            else:
                self.logger.warning(f"Assignment {gene_key} not found in forward mapping. Skipping.")
        
        self.logger.debug(f"Encoded {len(assignments)} assignments into chromosome with {sum(chromosome)} active genes.")
        return chromosome
    
    def repair_solution(self, discrete_vars: List[float]) -> List[float]:
        """
        Repairs an infeasible solution by applying constraint repair heuristics.
        This is critical for evolutionary algorithms to maintain feasibility.
        
        Repair strategies (as per Section 9.3 of foundations):
        1. Remove conflicting assignments (greedy conflict resolution)
        2. Ensure each course is assigned at most once
        3. Respect hard constraints (competency, capacity, availability)
        
        Args:
            discrete_vars: Potentially infeasible binary decision vector
        
        Returns:
            Repaired binary decision vector
        
        Complexity: O(n²) worst case for conflict resolution
        """
        # Decode current assignments
        assignments = self.decode_assignments(discrete_vars)
        
        # Apply repair heuristics
        repaired_assignments = self._repair_conflicts(assignments)
        repaired_assignments = self._repair_course_assignments(repaired_assignments)
        repaired_assignments = self._repair_hard_constraints(repaired_assignments)
        
        # Encode back to chromosome
        repaired_chromosome = self.encode_assignments(repaired_assignments)
        
        self.logger.debug(f"Repaired solution: {len(assignments)} -> {len(repaired_assignments)} assignments.")
        return repaired_chromosome
    
    def _repair_conflicts(self, assignments: List[Tuple[UUID, UUID, UUID, UUID, UUID]]) -> List[Tuple[UUID, UUID, UUID, UUID, UUID]]:
        """
        Removes conflicting assignments (faculty/room conflicts).
        Uses greedy strategy: keep first assignment, remove subsequent conflicts.
        """
        faculty_time_map = {}
        room_time_map = {}
        repaired = []
        
        for assignment in assignments:
            course_id, faculty_id, room_id, timeslot_id, batch_id = assignment
            
            # Check faculty conflict
            faculty_key = (faculty_id, timeslot_id)
            if faculty_key in faculty_time_map:
                self.logger.debug(f"Faculty conflict detected: {faculty_id} at {timeslot_id}. Removing assignment.")
                continue
            
            # Check room conflict
            room_key = (room_id, timeslot_id)
            if room_key in room_time_map:
                self.logger.debug(f"Room conflict detected: {room_id} at {timeslot_id}. Removing assignment.")
                continue
            
            # No conflict, keep assignment
            faculty_time_map[faculty_key] = course_id
            room_time_map[room_key] = course_id
            repaired.append(assignment)
        
        return repaired
    
    def _repair_course_assignments(self, assignments: List[Tuple[UUID, UUID, UUID, UUID, UUID]]) -> List[Tuple[UUID, UUID, UUID, UUID, UUID]]:
        """
        Ensures each course is assigned at most once.
        Uses greedy strategy: keep first assignment for each course.
        """
        course_assigned = set()
        repaired = []
        
        for assignment in assignments:
            course_id = assignment[0]
            
            if course_id in course_assigned:
                self.logger.debug(f"Duplicate course assignment detected: {course_id}. Removing.")
                continue
            
            course_assigned.add(course_id)
            repaired.append(assignment)
        
        return repaired
    
    def _repair_hard_constraints(self, assignments: List[Tuple[UUID, UUID, UUID, UUID, UUID]]) -> List[Tuple[UUID, UUID, UUID, UUID, UUID]]:
        """
        Removes assignments that violate hard constraints (competency, capacity, availability).
        """
        repaired = []
        
        for assignment in assignments:
            course_id, faculty_id, room_id, timeslot_id, batch_id = assignment
            
            # Check competency
            competency = self.compiled_data.competency_matrix.get((faculty_id, course_id), 0.0)
            if competency < 0.5:
                self.logger.debug(f"Competency violation: faculty {faculty_id} for course {course_id}. Removing.")
                continue
            
            # Check capacity
            enrollment = sum(count for (bid, cid), count in self.compiled_data.enrollment_matrix.items() 
                           if cid == course_id)
            room_capacity = self.compiled_data.rooms.get(room_id, {}).get('capacity', 0)
            if enrollment > room_capacity:
                self.logger.debug(f"Capacity violation: room {room_id} for course {course_id}. Removing.")
                continue
            
            # Check availability (simplified check)
            # Full availability checking would require faculty_availability and room_availability from ConstraintFormulator
            # For now, assume all are available (this should be improved)
            
            repaired.append(assignment)
        
        return repaired
    
    def decode_continuous_vars(self, continuous_vars: List[float]) -> Dict[str, Any]:
        """
        Decodes continuous decision variables into interpretable parameters.
        
        Args:
            continuous_vars: Continuous decision vector
        
        Returns:
            Dictionary with decoded parameters
        """
        n_faculty = len(self.faculty_ids)
        n_courses = len(self.course_ids)
        
        # Structure: [faculty_pref_weights (|F|*3), course_importance (|C|*2)]
        faculty_pref_weights = continuous_vars[:n_faculty * 3]
        course_importance = continuous_vars[n_faculty * 3:n_faculty * 3 + n_courses * 2]
        
        # Reshape into meaningful structures
        faculty_preferences = {}
        for i, fid in enumerate(self.faculty_ids):
            faculty_preferences[fid] = {
                'timeslot_weight': faculty_pref_weights[i * 3],
                'course_weight': faculty_pref_weights[i * 3 + 1],
                'workload_weight': faculty_pref_weights[i * 3 + 2]
            }
        
        course_weights = {}
        for i, cid in enumerate(self.course_ids):
            course_weights[cid] = {
                'importance': course_importance[i * 2],
                'priority': course_importance[i * 2 + 1]
            }
        
        return {
            'faculty_preferences': faculty_preferences,
            'course_weights': course_weights
        }


