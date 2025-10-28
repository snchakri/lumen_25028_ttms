"""
Metadata Extractor

Extract problem characteristics for solver selection.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
from typing import Dict, Any
from dataclasses import dataclass, field
import pandas as pd

from .loader import CompiledData
from .bijection import BijectiveMapper


@dataclass
class ProblemCharacteristics:
    """Problem characteristics for solver selection."""
    
    # Entity counts
    n_courses: int = 0
    n_faculty: int = 0
    n_rooms: int = 0
    n_timeslots: int = 0
    n_batches: int = 0
    
    # Problem size
    n_variables: int = 0
    n_hard_constraints: int = 0
    n_soft_constraints: int = 0
    n_constraints: int = 0
    
    # Complexity metrics
    constraint_density: float = 0.0
    problem_size_category: str = "small"  # small, medium, large
    sparsity_pattern: str = "unknown"
    
    # Additional metrics
    avg_courses_per_faculty: float = 0.0
    avg_students_per_batch: float = 0.0
    avg_capacity_utilization: float = 0.0


class MetadataExtractor:
    """
    Extract problem characteristics from compiled data.
    
    Used for solver selection (Algorithm 9.3).
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def extract(self, compiled_data: CompiledData, 
                bijective_mapper: BijectiveMapper) -> ProblemCharacteristics:
        """
        Extract problem characteristics.
        
        Returns:
            ProblemCharacteristics with all metrics
        """
        self.logger.info("Extracting problem characteristics")
        
        characteristics = ProblemCharacteristics()
        
        # Extract entity counts
        characteristics.n_courses = bijective_mapper.n_courses
        characteristics.n_faculty = bijective_mapper.n_faculty
        characteristics.n_rooms = bijective_mapper.n_rooms
        characteristics.n_timeslots = bijective_mapper.n_timeslots
        characteristics.n_batches = bijective_mapper.n_batches
        
        # Calculate problem size
        characteristics.n_variables = (
            characteristics.n_courses *
            characteristics.n_faculty *
            characteristics.n_rooms *
            characteristics.n_timeslots *
            characteristics.n_batches
        )
        
        # Estimate constraint counts (simplified)
        # Hard constraints: faculty conflicts + room conflicts + batch conflicts
        characteristics.n_hard_constraints = (
            characteristics.n_timeslots * characteristics.n_faculty +  # faculty conflicts
            characteristics.n_timeslots * characteristics.n_rooms +    # room conflicts
            characteristics.n_timeslots * characteristics.n_batches +  # batch conflicts
            characteristics.n_courses  # course coverage
        )
        
        # Soft constraints: preferences, workload balance, etc.
        characteristics.n_soft_constraints = (
            characteristics.n_faculty * characteristics.n_courses +  # course preferences
            characteristics.n_faculty * characteristics.n_timeslots   # time preferences
        )
        
        characteristics.n_constraints = (
            characteristics.n_hard_constraints + 
            characteristics.n_soft_constraints
        )
        
        # Calculate constraint density
        if characteristics.n_variables > 0:
            characteristics.constraint_density = (
                characteristics.n_constraints / (characteristics.n_variables ** 2)
            )
        
        # Classify problem size
        if characteristics.n_variables < 10000:
            characteristics.problem_size_category = "small"
        elif characteristics.n_variables < 100000:
            characteristics.problem_size_category = "medium"
        else:
            characteristics.problem_size_category = "large"
        
        # Calculate additional metrics
        if characteristics.n_faculty > 0:
            characteristics.avg_courses_per_faculty = (
                characteristics.n_courses / characteristics.n_faculty
            )
        
        # Extract batch sizes
        batches_df = compiled_data.L_raw.get('student_batches', pd.DataFrame())
        if not batches_df.empty and 'student_count' in batches_df.columns:
            characteristics.avg_students_per_batch = batches_df['student_count'].mean()
        
        # Extract room utilization
        rooms_df = compiled_data.L_raw.get('rooms', pd.DataFrame())
        if not rooms_df.empty and 'capacity' in rooms_df.columns:
            total_capacity = rooms_df['capacity'].sum()
            if total_capacity > 0 and characteristics.n_batches > 0:
                total_students = characteristics.avg_students_per_batch * characteristics.n_batches
                characteristics.avg_capacity_utilization = total_students / total_capacity
        
        self.logger.info(f"Problem characteristics extracted:")
        self.logger.info(f"  - Variables: {characteristics.n_variables}")
        self.logger.info(f"  - Constraints: {characteristics.n_constraints}")
        self.logger.info(f"  - Constraint density: {characteristics.constraint_density:.4f}")
        self.logger.info(f"  - Problem size: {characteristics.problem_size_category}")
        
        return characteristics


Metadata Extractor

Extract problem characteristics for solver selection.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
from typing import Dict, Any
from dataclasses import dataclass, field
import pandas as pd

from .loader import CompiledData
from .bijection import BijectiveMapper


@dataclass
class ProblemCharacteristics:
    """Problem characteristics for solver selection."""
    
    # Entity counts
    n_courses: int = 0
    n_faculty: int = 0
    n_rooms: int = 0
    n_timeslots: int = 0
    n_batches: int = 0
    
    # Problem size
    n_variables: int = 0
    n_hard_constraints: int = 0
    n_soft_constraints: int = 0
    n_constraints: int = 0
    
    # Complexity metrics
    constraint_density: float = 0.0
    problem_size_category: str = "small"  # small, medium, large
    sparsity_pattern: str = "unknown"
    
    # Additional metrics
    avg_courses_per_faculty: float = 0.0
    avg_students_per_batch: float = 0.0
    avg_capacity_utilization: float = 0.0


class MetadataExtractor:
    """
    Extract problem characteristics from compiled data.
    
    Used for solver selection (Algorithm 9.3).
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def extract(self, compiled_data: CompiledData, 
                bijective_mapper: BijectiveMapper) -> ProblemCharacteristics:
        """
        Extract problem characteristics.
        
        Returns:
            ProblemCharacteristics with all metrics
        """
        self.logger.info("Extracting problem characteristics")
        
        characteristics = ProblemCharacteristics()
        
        # Extract entity counts
        characteristics.n_courses = bijective_mapper.n_courses
        characteristics.n_faculty = bijective_mapper.n_faculty
        characteristics.n_rooms = bijective_mapper.n_rooms
        characteristics.n_timeslots = bijective_mapper.n_timeslots
        characteristics.n_batches = bijective_mapper.n_batches
        
        # Calculate problem size
        characteristics.n_variables = (
            characteristics.n_courses *
            characteristics.n_faculty *
            characteristics.n_rooms *
            characteristics.n_timeslots *
            characteristics.n_batches
        )
        
        # Estimate constraint counts (simplified)
        # Hard constraints: faculty conflicts + room conflicts + batch conflicts
        characteristics.n_hard_constraints = (
            characteristics.n_timeslots * characteristics.n_faculty +  # faculty conflicts
            characteristics.n_timeslots * characteristics.n_rooms +    # room conflicts
            characteristics.n_timeslots * characteristics.n_batches +  # batch conflicts
            characteristics.n_courses  # course coverage
        )
        
        # Soft constraints: preferences, workload balance, etc.
        characteristics.n_soft_constraints = (
            characteristics.n_faculty * characteristics.n_courses +  # course preferences
            characteristics.n_faculty * characteristics.n_timeslots   # time preferences
        )
        
        characteristics.n_constraints = (
            characteristics.n_hard_constraints + 
            characteristics.n_soft_constraints
        )
        
        # Calculate constraint density
        if characteristics.n_variables > 0:
            characteristics.constraint_density = (
                characteristics.n_constraints / (characteristics.n_variables ** 2)
            )
        
        # Classify problem size
        if characteristics.n_variables < 10000:
            characteristics.problem_size_category = "small"
        elif characteristics.n_variables < 100000:
            characteristics.problem_size_category = "medium"
        else:
            characteristics.problem_size_category = "large"
        
        # Calculate additional metrics
        if characteristics.n_faculty > 0:
            characteristics.avg_courses_per_faculty = (
                characteristics.n_courses / characteristics.n_faculty
            )
        
        # Extract batch sizes
        batches_df = compiled_data.L_raw.get('student_batches', pd.DataFrame())
        if not batches_df.empty and 'student_count' in batches_df.columns:
            characteristics.avg_students_per_batch = batches_df['student_count'].mean()
        
        # Extract room utilization
        rooms_df = compiled_data.L_raw.get('rooms', pd.DataFrame())
        if not rooms_df.empty and 'capacity' in rooms_df.columns:
            total_capacity = rooms_df['capacity'].sum()
            if total_capacity > 0 and characteristics.n_batches > 0:
                total_students = characteristics.avg_students_per_batch * characteristics.n_batches
                characteristics.avg_capacity_utilization = total_students / total_capacity
        
        self.logger.info(f"Problem characteristics extracted:")
        self.logger.info(f"  - Variables: {characteristics.n_variables}")
        self.logger.info(f"  - Constraints: {characteristics.n_constraints}")
        self.logger.info(f"  - Constraint density: {characteristics.constraint_density:.4f}")
        self.logger.info(f"  - Problem size: {characteristics.problem_size_category}")
        
        return characteristics




