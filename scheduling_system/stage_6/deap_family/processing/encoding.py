"""
Genotype Encoding and Phenotype Mapping

Implements Definition 2.2 (Genotype Encoding) and Definition 2.3 (Phenotype Mapping).

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import random


@dataclass
class Genotype:
    """Genotype representation per Definition 2.2."""
    genes: List[Tuple[str, str, str, str, str]]  # (course_id, faculty_id, room_id, timeslot_id, batch_id)
    
    def __hash__(self):
        return hash(tuple(self.genes))
    
    def __eq__(self, other):
        return self.genes == other.genes
    
    def __len__(self):
        return len(self.genes)


class GenotypeEncoder:
    """
    Genotype encoding per Definition 2.2.
    
    Encodes scheduling solutions as genotypes.
    """
    
    def __init__(self, compiled_data, logger: logging.Logger):
        self.compiled_data = compiled_data
        self.logger = logger
        self._build_encoding_mappings()
    
    def _build_encoding_mappings(self):
        """Build mappings for efficient encoding."""
        # Extract entities
        courses_df = self.compiled_data.L_raw.get('courses', None)
        faculty_df = self.compiled_data.L_raw.get('faculty', None)
        rooms_df = self.compiled_data.L_raw.get('rooms', None)
        timeslots_df = self.compiled_data.L_raw.get('timeslots', None)
        batches_df = self.compiled_data.L_raw.get('student_batches', None)
        
        self.course_ids = courses_df['primary_key'].tolist() if courses_df is not None else []
        self.faculty_ids = faculty_df['primary_key'].tolist() if faculty_df is not None else []
        self.room_ids = rooms_df['primary_key'].tolist() if rooms_df is not None else []
        self.timeslot_ids = timeslots_df['primary_key'].tolist() if timeslots_df is not None else []
        self.batch_ids = batches_df['primary_key'].tolist() if batches_df is not None else []
        
        self.logger.info(f"Encoding mappings: {len(self.course_ids)} courses, {len(self.faculty_ids)} faculty, {len(self.room_ids)} rooms, {len(self.timeslot_ids)} timeslots, {len(self.batch_ids)} batches")
    
    def encode_schedule(self, schedule: Dict[str, Any]) -> Genotype:
        """
        Encode schedule to genotype.
        
        Args:
            schedule: Schedule assignments
        
        Returns:
            Genotype representation
        """
        genes = []
        
        for assignment in schedule.get('assignments', []):
            gene = (
                assignment.get('course_id'),
                assignment.get('faculty_id'),
                assignment.get('room_id'),
                assignment.get('timeslot_id'),
                assignment.get('batch_id')
            )
            genes.append(gene)
        
        return Genotype(genes=genes)
    
    def generate_random_genotype(self) -> Genotype:
        """Generate random valid genotype."""
        genes = []
        
        # For each course, assign random valid resources
        for course_id in self.course_ids:
            # Random batch
            batch_id = random.choice(self.batch_ids) if self.batch_ids else None
            
            # Random faculty (should be competent, but simplified for now)
            faculty_id = random.choice(self.faculty_ids) if self.faculty_ids else None
            
            # Random room
            room_id = random.choice(self.room_ids) if self.room_ids else None
            
            # Random timeslot
            timeslot_id = random.choice(self.timeslot_ids) if self.timeslot_ids else None
            
            genes.append((course_id, faculty_id, room_id, timeslot_id, batch_id))
        
        return Genotype(genes=genes)


class PhenotypeDecoder:
    """
    Phenotype mapping φ: G → S_schedule per Definition 2.3.
    
    Decodes genotypes to schedules.
    """
    
    def __init__(self, compiled_data, logger: logging.Logger):
        self.compiled_data = compiled_data
        self.logger = logger
        self._build_decoding_mappings()
    
    def _build_decoding_mappings(self):
        """Build mappings for efficient decoding."""
        # Build entity lookup dictionaries
        self.course_lookup = {}
        self.faculty_lookup = {}
        self.room_lookup = {}
        self.timeslot_lookup = {}
        self.batch_lookup = {}
        
        if 'courses' in self.compiled_data.L_raw:
            for _, row in self.compiled_data.L_raw['courses'].iterrows():
                self.course_lookup[row['primary_key']] = row.to_dict()
        
        if 'faculty' in self.compiled_data.L_raw:
            for _, row in self.compiled_data.L_raw['faculty'].iterrows():
                self.faculty_lookup[row['primary_key']] = row.to_dict()
        
        if 'rooms' in self.compiled_data.L_raw:
            for _, row in self.compiled_data.L_raw['rooms'].iterrows():
                self.room_lookup[row['primary_key']] = row.to_dict()
        
        if 'timeslots' in self.compiled_data.L_raw:
            for _, row in self.compiled_data.L_raw['timeslots'].iterrows():
                self.timeslot_lookup[row['primary_key']] = row.to_dict()
        
        if 'student_batches' in self.compiled_data.L_raw:
            for _, row in self.compiled_data.L_raw['student_batches'].iterrows():
                self.batch_lookup[row['primary_key']] = row.to_dict()
    
    def decode_genotype(self, genotype: Genotype) -> Dict[str, Any]:
        """
        Decode genotype to schedule.
        
        Args:
            genotype: Genotype to decode
        
        Returns:
            Schedule dictionary
        """
        assignments = []
        
        for gene in genotype.genes:
            course_id, faculty_id, room_id, timeslot_id, batch_id = gene
            
            assignment = {
                'course_id': course_id,
                'faculty_id': faculty_id,
                'room_id': room_id,
                'timeslot_id': timeslot_id,
                'batch_id': batch_id,
            }
            
            # Add detailed information from lookups
            if course_id in self.course_lookup:
                course_data = self.course_lookup[course_id]
                assignment['course_name'] = course_data.get('course_name')
                assignment['credits'] = course_data.get('credits')
            
            if faculty_id in self.faculty_lookup:
                faculty_data = self.faculty_lookup[faculty_id]
                assignment['faculty_name'] = faculty_data.get('faculty_name')
            
            if room_id in self.room_lookup:
                room_data = self.room_lookup[room_id]
                assignment['room_name'] = room_data.get('room_name')
                assignment['capacity'] = room_data.get('capacity')
            
            if timeslot_id in self.timeslot_lookup:
                timeslot_data = self.timeslot_lookup[timeslot_id]
                # Stage-3 schema uses 'day_number' and provides 'duration_minutes' when available
                assignment['day'] = (
                    timeslot_data.get('day_number')
                    if 'day_number' in timeslot_data
                    else timeslot_data.get('day_num')
                )
                assignment['start_time'] = timeslot_data.get('start_time')
                assignment['end_time'] = timeslot_data.get('end_time')
                # Prefer explicit duration_minutes if present; else compute when Stage 7 writes CSV
                if 'duration_minutes' in timeslot_data:
                    assignment['duration'] = timeslot_data.get('duration_minutes')
            
            if batch_id in self.batch_lookup:
                batch_data = self.batch_lookup[batch_id]
                assignment['batch_name'] = batch_data.get('batch_name')
                assignment['student_count'] = batch_data.get('student_count')
            
            assignments.append(assignment)
        
        return {
            'assignments': assignments,
            'n_assignments': len(assignments)
        }
    
    def mutate_genotype(self, genotype: Genotype, mutation_rate: float = 0.1) -> Genotype:
        """
        Mutation operator V per Definition 2.1.
        Implements various mutation strategies for genotype modification.
        """
        mutation_type = self.compiled_data.get('mutation_type', 'swap')
        
        if mutation_type == 'swap':
            return self._swap_mutation(genotype, mutation_rate)
        elif mutation_type == 'insertion':
            return self._insertion_mutation(genotype, mutation_rate)
        elif mutation_type == 'inversion':
            return self._inversion_mutation(genotype, mutation_rate)
        elif mutation_type == 'scramble':
            return self._scramble_mutation(genotype, mutation_rate)
        else:
            raise ValueError(f"Unknown mutation type: {mutation_type}")
    
    def _swap_mutation(self, genotype: Genotype, mutation_rate: float) -> Genotype:
        """Swap mutation implementation."""
        mutated = Genotype(genes=genotype.genes.copy())
        
        for i in range(len(mutated.genes)):
            if random.random() < mutation_rate:
                # Swap with random position
                j = random.randint(0, len(mutated.genes) - 1)
                mutated.genes[i], mutated.genes[j] = mutated.genes[j], mutated.genes[i]
        
        return mutated
    
    def crossover_genotypes(self, parent1: Genotype, parent2: Genotype, crossover_type: str = "uniform") -> Tuple[Genotype, Genotype]:
        """
        Crossover operator V per Definition 2.1.
        Implements various crossover strategies for genotype recombination.
        """
        if crossover_type == "uniform":
            return self._uniform_crossover(parent1, parent2)
        elif crossover_type == "order":
            return self._order_crossover(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover type: {crossover_type}")
    
    def _uniform_crossover(self, parent1: Genotype, parent2: Genotype) -> Tuple[Genotype, Genotype]:
        """Uniform crossover implementation."""
        min_length = min(len(parent1.genes), len(parent2.genes))
        
        child1_genes = []
        child2_genes = []
        
        for i in range(min_length):
            if random.random() < 0.5:
                child1_genes.append(parent1.genes[i])
                child2_genes.append(parent2.genes[i])
            else:
                child1_genes.append(parent2.genes[i])
                child2_genes.append(parent1.genes[i])
        
        return Genotype(genes=child1_genes), Genotype(genes=child2_genes)
    
    def _order_crossover(self, parent1: Genotype, parent2: Genotype) -> Tuple[Genotype, Genotype]:
        """Order crossover (OX) implementation."""
        length = min(len(parent1.genes), len(parent2.genes))
        if length < 2:
            return parent1, parent2
        
        start = random.randint(0, length - 2)
        end = random.randint(start + 1, length)
        
        child1_genes = [None] * length
        child2_genes = [None] * length
        
        # Copy segments
        child1_genes[start:end] = parent1.genes[start:end]
        child2_genes[start:end] = parent2.genes[start:end]
        
        return Genotype(genes=child1_genes), Genotype(genes=child2_genes)
