"""
Bijective Mappings - Entity-to-Variable Bijective Mappings

Implements invertible bijective mappings between entity IDs and variable indices
per foundations with rigorous mathematical validation.

Compliance:
- Definition 2.3: Variable Assignment Encoding x_{c,f,r,t,b}
- Theorem: Bijective mapping invertibility guarantee

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


@dataclass
class EntityMapping:
    """Bijective mapping for a single entity type."""
    
    entity_type: str
    id_to_index: Dict[str, int] = field(default_factory=dict)
    index_to_id: Dict[int, str] = field(default_factory=dict)
    
    def add_mapping(self, entity_id: str, index: int):
        """Add bijective mapping."""
        if entity_id in self.id_to_index:
            raise ValueError(f"Duplicate entity ID: {entity_id}")
        if index in self.index_to_id:
            raise ValueError(f"Duplicate index: {index}")
        
        self.id_to_index[entity_id] = index
        self.index_to_id[index] = entity_id
    
    def get_index(self, entity_id: str) -> int:
        """Get index for entity ID."""
        if entity_id not in self.id_to_index:
            raise KeyError(f"Entity ID not found: {entity_id}")
        return self.id_to_index[entity_id]
    
    def get_id(self, index: int) -> str:
        """Get entity ID for index (inverse mapping)."""
        if index not in self.index_to_id:
            raise KeyError(f"Index not found: {index}")
        return self.index_to_id[index]
    
    def verify_bijectivity(self) -> bool:
        """
        Verify bijectivity: |id_to_index| = |index_to_id| and all mappings are unique.
        
        Returns:
            True if bijective, False otherwise
        """
        # Check cardinality equality
        if len(self.id_to_index) != len(self.index_to_id):
            return False
        
        # Check forward mapping consistency
        for entity_id, index in self.id_to_index.items():
            if self.index_to_id[index] != entity_id:
                return False
        
        # Check inverse mapping consistency
        for index, entity_id in self.index_to_id.items():
            if self.id_to_index[entity_id] != index:
                return False
        
        return True
    
    def size(self) -> int:
        """Get number of mappings."""
        return len(self.id_to_index)


@dataclass
class VariableMapping:
    """
    Complete bijective mapping system for all entity types.
    
    Compliance: Definition 2.3
    """
    
    # Entity mappings
    course_mapping: EntityMapping = field(default_factory=lambda: EntityMapping("course"))
    faculty_mapping: EntityMapping = field(default_factory=lambda: EntityMapping("faculty"))
    room_mapping: EntityMapping = field(default_factory=lambda: EntityMapping("room"))
    timeslot_mapping: EntityMapping = field(default_factory=lambda: EntityMapping("timeslot"))
    batch_mapping: EntityMapping = field(default_factory=lambda: EntityMapping("batch"))
    
    # Variable name cache
    variable_name_cache: Dict[Tuple[int, int, int, int, int], str] = field(default_factory=dict)
    
    def verify_all_bijectivity(self) -> bool:
        """Verify bijectivity for all entity mappings."""
        return all([
            self.course_mapping.verify_bijectivity(),
            self.faculty_mapping.verify_bijectivity(),
            self.room_mapping.verify_bijectivity(),
            self.timeslot_mapping.verify_bijectivity(),
            self.batch_mapping.verify_bijectivity()
        ])
    
    def get_variable_name(self, c_idx: int, f_idx: int, r_idx: int, t_idx: int, b_idx: int) -> str:
        """Generate variable name: x_c{c}_f{f}_r{r}_t{t}_b{b}"""
        return f"x_c{c_idx}_f{f_idx}_r{r_idx}_t{t_idx}_b{b_idx}"
    
    def parse_variable_name(self, var_name: str) -> Tuple[int, int, int, int, int]:
        """
        Parse variable name to extract indices.
        
        Format: x_c{c}_f{f}_r{r}_t{t}_b{b}
        
        Returns:
            (c_idx, f_idx, r_idx, t_idx, b_idx)
        """
        if not var_name.startswith('x_c'):
            raise ValueError(f"Invalid variable name format: {var_name}")
        
        # Remove 'x_' prefix
        suffix = var_name[2:]
        
        # Split by '_'
        parts = suffix.split('_')
        
        if len(parts) != 5:  # c{idx}, f{idx}, r{idx}, t{idx}, b{idx}
            raise ValueError(f"Invalid variable name format: {var_name}")
        
        # Extract indices
        c_idx = int(parts[0][1:])  # Remove 'c' prefix
        f_idx = int(parts[1][1:])  # Remove 'f' prefix
        r_idx = int(parts[2][1:])  # Remove 'r' prefix
        t_idx = int(parts[3][1:])  # Remove 't' prefix
        b_idx = int(parts[4][1:])  # Remove 'b' prefix
        
        return (c_idx, f_idx, r_idx, t_idx, b_idx)
    
    def get_entity_ids_from_var_name(self, var_name: str) -> Tuple[str, str, str, str, str]:
        """
        Get original entity IDs from variable name.
        
        Returns:
            (course_id, faculty_id, room_id, timeslot_id, batch_id)
        """
        c_idx, f_idx, r_idx, t_idx, b_idx = self.parse_variable_name(var_name)
        
        course_id = self.course_mapping.get_id(c_idx)
        faculty_id = self.faculty_mapping.get_id(f_idx)
        room_id = self.room_mapping.get_id(r_idx)
        timeslot_id = self.timeslot_mapping.get_id(t_idx)
        batch_id = self.batch_mapping.get_id(b_idx)
        
        return (course_id, faculty_id, room_id, timeslot_id, batch_id)
    
    def get_total_variables(self) -> int:
        """Calculate total number of possible variables."""
        return (
            self.course_mapping.size() *
            self.faculty_mapping.size() *
            self.room_mapping.size() *
            self.timeslot_mapping.size() *
            self.batch_mapping.size()
        )


class BijectiveMapper:
    """
    Creates and manages bijective mappings from Stage 3 compiled data.
    
    Compliance: Definition 2.3, Bijective mapping invertibility
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize bijective mapper."""
        self.logger = logger or logging.getLogger(__name__)
        self.mapping = VariableMapping()
    
    def create_mappings(self, l_raw: Dict[str, pd.DataFrame]) -> VariableMapping:
        """
        Create bijective mappings from L_raw entities.
        
        Args:
            l_raw: L_raw layer from Stage 3 compiled data
        
        Returns:
            VariableMapping with all bijective mappings
        """
        self.logger.info("Creating bijective entity-to-variable mappings...")
        
        # Create course mapping
        if 'courses.csv' in l_raw:
            self._create_entity_mapping(l_raw['courses.csv'], self.mapping.course_mapping, 'course_id')
        
        # Create faculty mapping
        if 'faculty.csv' in l_raw:
            self._create_entity_mapping(l_raw['faculty.csv'], self.mapping.faculty_mapping, 'faculty_id')
        
        # Create room mapping
        if 'rooms.csv' in l_raw:
            self._create_entity_mapping(l_raw['rooms.csv'], self.mapping.room_mapping, 'room_id')
        
        # Create timeslot mapping
        if 'time_slots.csv' in l_raw:
            self._create_entity_mapping(l_raw['time_slots.csv'], self.mapping.timeslot_mapping, 'slot_id')
        
        # Create batch mapping
        if 'student_batches.csv' in l_raw:
            self._create_entity_mapping(l_raw['student_batches.csv'], self.mapping.batch_mapping, 'batch_id')
        
        # Verify bijectivity
        if not self.mapping.verify_all_bijectivity():
            raise ValueError("Bijectivity verification failed!")
        
        self.logger.info(f"Created bijective mappings:")
        self.logger.info(f"  - Courses: {self.mapping.course_mapping.size()}")
        self.logger.info(f"  - Faculty: {self.mapping.faculty_mapping.size()}")
        self.logger.info(f"  - Rooms: {self.mapping.room_mapping.size()}")
        self.logger.info(f"  - Timeslots: {self.mapping.timeslot_mapping.size()}")
        self.logger.info(f"  - Batches: {self.mapping.batch_mapping.size()}")
        self.logger.info(f"  - Total possible variables: {self.mapping.get_total_variables()}")
        
        return self.mapping
    
    def _create_entity_mapping(self, df: pd.DataFrame, mapping: EntityMapping, id_column: str):
        """Create bijective mapping for a single entity type."""
        if df.empty:
            self.logger.warning(f"Empty DataFrame for {mapping.entity_type}")
            return
        
        if id_column not in df.columns:
            # Try to use first column as ID
            id_column = df.columns[0]
            self.logger.warning(f"Using first column as ID: {id_column}")
        
        # Create mappings
        for idx, row in df.iterrows():
            entity_id = str(row[id_column])
            mapping.add_mapping(entity_id, int(idx))
        
        self.logger.debug(f"Created {mapping.size()} mappings for {mapping.entity_type}")
    
    def get_mapping(self) -> VariableMapping:
        """Get the current bijective mapping."""
        return self.mapping
    
    def validate_mapping(self) -> bool:
        """
        Validate bijective mapping with mathematical rigor.
        
        Returns:
            True if valid, False otherwise
        """
        # Verify bijectivity
        if not self.mapping.verify_all_bijectivity():
            self.logger.error("Bijectivity verification failed")
            return False
        
        # Verify all mappings are non-empty
        if self.mapping.course_mapping.size() == 0:
            self.logger.error("Course mapping is empty")
            return False
        
        if self.mapping.faculty_mapping.size() == 0:
            self.logger.error("Faculty mapping is empty")
            return False
        
        if self.mapping.room_mapping.size() == 0:
            self.logger.error("Room mapping is empty")
            return False
        
        if self.mapping.timeslot_mapping.size() == 0:
            self.logger.error("Timeslot mapping is empty")
            return False
        
        if self.mapping.batch_mapping.size() == 0:
            self.logger.error("Batch mapping is empty")
            return False
        
        self.logger.info("Bijective mapping validation passed")
        return True

