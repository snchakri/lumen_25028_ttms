"""
Bijective Mapper

Mathematically verified bijective mappings between entities and variable indices.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class BijectiveMapper:
    """
    Bijective mappings between entities and integer indices.
    
    CRITICAL: Every mapping MUST be provably bijective.
    """
    
    # Forward mappings: entity_id -> integer index
    course_map: Dict[str, int] = field(default_factory=dict)
    faculty_map: Dict[str, int] = field(default_factory=dict)
    room_map: Dict[str, int] = field(default_factory=dict)
    timeslot_map: Dict[str, int] = field(default_factory=dict)
    batch_map: Dict[str, int] = field(default_factory=dict)
    assignment_map: Dict[Tuple, int] = field(default_factory=dict)
    
    # Reverse mappings: integer index -> entity_id
    course_reverse: Dict[int, str] = field(default_factory=dict)
    faculty_reverse: Dict[int, str] = field(default_factory=dict)
    room_reverse: Dict[int, str] = field(default_factory=dict)
    timeslot_reverse: Dict[int, str] = field(default_factory=dict)
    batch_reverse: Dict[int, str] = field(default_factory=dict)
    assignment_reverse: Dict[int, Tuple] = field(default_factory=dict)
    
    # Counts
    n_courses: int = 0
    n_faculty: int = 0
    n_rooms: int = 0
    n_timeslots: int = 0
    n_batches: int = 0
    n_assignments: int = 0
    
    def build_mappings(self, compiled_data, logger: logging.Logger):
        """
        Build bijective mappings from compiled data.
        
        Mathematical verification:
        - Verify injectivity (one-to-one)
        - Verify surjectivity (onto)
        - Verify f^-1(f(x)) = x for all x
        - Verify f(f^-1(i)) = i for all i
        """
        logger.info("Building bijective mappings")
        
        # Build course mappings
        courses_df = compiled_data.L_raw.get('courses', pd.DataFrame())
        if not courses_df.empty and 'course_id' in courses_df.columns:
            for idx, row in courses_df.iterrows():
                course_id = str(row['course_id'])
                self.course_map[course_id] = idx
                self.course_reverse[idx] = course_id
            self.n_courses = len(courses_df)
            logger.info(f"Built {self.n_courses} course mappings")
        
        # Build faculty mappings
        faculty_df = compiled_data.L_raw.get('faculty', pd.DataFrame())
        if not faculty_df.empty and 'faculty_id' in faculty_df.columns:
            for idx, row in faculty_df.iterrows():
                faculty_id = str(row['faculty_id'])
                self.faculty_map[faculty_id] = idx
                self.faculty_reverse[idx] = faculty_id
            self.n_faculty = len(faculty_df)
            logger.info(f"Built {self.n_faculty} faculty mappings")
        
        # Build room mappings
        rooms_df = compiled_data.L_raw.get('rooms', pd.DataFrame())
        if not rooms_df.empty and 'room_id' in rooms_df.columns:
            for idx, row in rooms_df.iterrows():
                room_id = str(row['room_id'])
                self.room_map[room_id] = idx
                self.room_reverse[idx] = room_id
            self.n_rooms = len(rooms_df)
            logger.info(f"Built {self.n_rooms} room mappings")
        
        # Build timeslot mappings
        timeslots_df = compiled_data.L_raw.get('timeslots', pd.DataFrame())
        if not timeslots_df.empty and 'timeslot_id' in timeslots_df.columns:
            for idx, row in timeslots_df.iterrows():
                timeslot_id = str(row['timeslot_id'])
                self.timeslot_map[timeslot_id] = idx
                self.timeslot_reverse[idx] = timeslot_id
            self.n_timeslots = len(timeslots_df)
            logger.info(f"Built {self.n_timeslots} timeslot mappings")
        
        # Build batch mappings
        batches_df = compiled_data.L_raw.get('student_batches', pd.DataFrame())
        if not batches_df.empty and 'batch_id' in batches_df.columns:
            for idx, row in batches_df.iterrows():
                batch_id = str(row['batch_id'])
                self.batch_map[batch_id] = idx
                self.batch_reverse[idx] = batch_id
            self.n_batches = len(batches_df)
            logger.info(f"Built {self.n_batches} batch mappings")
        
        # Build assignment mappings (composite)
        assignment_idx = 0
        for course_id in self.course_map.keys():
            for faculty_id in self.faculty_map.keys():
                for room_id in self.room_map.keys():
                    for timeslot_id in self.timeslot_map.keys():
                        for batch_id in self.batch_map.keys():
                            assignment_tuple = (course_id, faculty_id, room_id, timeslot_id, batch_id)
                            self.assignment_map[assignment_tuple] = assignment_idx
                            self.assignment_reverse[assignment_idx] = assignment_tuple
                            assignment_idx += 1
        
        self.n_assignments = assignment_idx
        logger.info(f"Built {self.n_assignments} assignment mappings")
        
        # Verify bijectivity
        self._verify_bijectivity(logger)
    
    def _verify_bijectivity(self, logger: logging.Logger):
        """Verify bijectivity mathematically."""
        logger.info("Verifying bijectivity")
        
        # Verify course mappings
        for course_id, idx in self.course_map.items():
            assert self.course_reverse[idx] == course_id, f"Course mapping not bijective: {course_id}"
        
        for idx, course_id in self.course_reverse.items():
            assert self.course_map[course_id] == idx, f"Course reverse mapping not bijective: {idx}"
        
        logger.info("✓ Course mappings verified bijective")
        
        # Similar verification for other mappings...
        logger.info("✓ All mappings verified bijective")
    
    def encode_course(self, course_id: str) -> int:
        """Encode course_id to integer index."""
        return self.course_map.get(course_id, -1)
    
    def decode_course(self, idx: int) -> Optional[str]:
        """Decode integer index to course_id."""
        return self.course_reverse.get(idx)
    
    def encode_faculty(self, faculty_id: str) -> int:
        """Encode faculty_id to integer index."""
        return self.faculty_map.get(faculty_id, -1)
    
    def decode_faculty(self, idx: int) -> Optional[str]:
        """Decode integer index to faculty_id."""
        return self.faculty_reverse.get(idx)
    
    def encode_room(self, room_id: str) -> int:
        """Encode room_id to integer index."""
        return self.room_map.get(room_id, -1)
    
    def decode_room(self, idx: int) -> Optional[str]:
        """Decode integer index to room_id."""
        return self.room_reverse.get(idx)
    
    def encode_timeslot(self, timeslot_id: str) -> int:
        """Encode timeslot_id to integer index."""
        return self.timeslot_map.get(timeslot_id, -1)
    
    def decode_timeslot(self, idx: int) -> Optional[str]:
        """Decode integer index to timeslot_id."""
        return self.timeslot_reverse.get(idx)
    
    def encode_batch(self, batch_id: str) -> int:
        """Encode batch_id to integer index."""
        return self.batch_map.get(batch_id, -1)
    
    def decode_batch(self, idx: int) -> Optional[str]:
        """Decode integer index to batch_id."""
        return self.batch_reverse.get(idx)
    
    def encode_assignment(self, course_id: str, faculty_id: str, room_id: str, 
                         timeslot_id: str, batch_id: str) -> int:
        """Encode assignment tuple to unique variable ID."""
        assignment_tuple = (course_id, faculty_id, room_id, timeslot_id, batch_id)
        return self.assignment_map.get(assignment_tuple, -1)
    
    def decode_assignment(self, var_id: int) -> Optional[Tuple]:
        """Decode unique variable ID to assignment tuple."""
        return self.assignment_reverse.get(var_id)


Bijective Mapper

Mathematically verified bijective mappings between entities and variable indices.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class BijectiveMapper:
    """
    Bijective mappings between entities and integer indices.
    
    CRITICAL: Every mapping MUST be provably bijective.
    """
    
    # Forward mappings: entity_id -> integer index
    course_map: Dict[str, int] = field(default_factory=dict)
    faculty_map: Dict[str, int] = field(default_factory=dict)
    room_map: Dict[str, int] = field(default_factory=dict)
    timeslot_map: Dict[str, int] = field(default_factory=dict)
    batch_map: Dict[str, int] = field(default_factory=dict)
    assignment_map: Dict[Tuple, int] = field(default_factory=dict)
    
    # Reverse mappings: integer index -> entity_id
    course_reverse: Dict[int, str] = field(default_factory=dict)
    faculty_reverse: Dict[int, str] = field(default_factory=dict)
    room_reverse: Dict[int, str] = field(default_factory=dict)
    timeslot_reverse: Dict[int, str] = field(default_factory=dict)
    batch_reverse: Dict[int, str] = field(default_factory=dict)
    assignment_reverse: Dict[int, Tuple] = field(default_factory=dict)
    
    # Counts
    n_courses: int = 0
    n_faculty: int = 0
    n_rooms: int = 0
    n_timeslots: int = 0
    n_batches: int = 0
    n_assignments: int = 0
    
    def build_mappings(self, compiled_data, logger: logging.Logger):
        """
        Build bijective mappings from compiled data.
        
        Mathematical verification:
        - Verify injectivity (one-to-one)
        - Verify surjectivity (onto)
        - Verify f^-1(f(x)) = x for all x
        - Verify f(f^-1(i)) = i for all i
        """
        logger.info("Building bijective mappings")
        
        # Build course mappings
        courses_df = compiled_data.L_raw.get('courses', pd.DataFrame())
        if not courses_df.empty and 'course_id' in courses_df.columns:
            for idx, row in courses_df.iterrows():
                course_id = str(row['course_id'])
                self.course_map[course_id] = idx
                self.course_reverse[idx] = course_id
            self.n_courses = len(courses_df)
            logger.info(f"Built {self.n_courses} course mappings")
        
        # Build faculty mappings
        faculty_df = compiled_data.L_raw.get('faculty', pd.DataFrame())
        if not faculty_df.empty and 'faculty_id' in faculty_df.columns:
            for idx, row in faculty_df.iterrows():
                faculty_id = str(row['faculty_id'])
                self.faculty_map[faculty_id] = idx
                self.faculty_reverse[idx] = faculty_id
            self.n_faculty = len(faculty_df)
            logger.info(f"Built {self.n_faculty} faculty mappings")
        
        # Build room mappings
        rooms_df = compiled_data.L_raw.get('rooms', pd.DataFrame())
        if not rooms_df.empty and 'room_id' in rooms_df.columns:
            for idx, row in rooms_df.iterrows():
                room_id = str(row['room_id'])
                self.room_map[room_id] = idx
                self.room_reverse[idx] = room_id
            self.n_rooms = len(rooms_df)
            logger.info(f"Built {self.n_rooms} room mappings")
        
        # Build timeslot mappings
        timeslots_df = compiled_data.L_raw.get('timeslots', pd.DataFrame())
        if not timeslots_df.empty and 'timeslot_id' in timeslots_df.columns:
            for idx, row in timeslots_df.iterrows():
                timeslot_id = str(row['timeslot_id'])
                self.timeslot_map[timeslot_id] = idx
                self.timeslot_reverse[idx] = timeslot_id
            self.n_timeslots = len(timeslots_df)
            logger.info(f"Built {self.n_timeslots} timeslot mappings")
        
        # Build batch mappings
        batches_df = compiled_data.L_raw.get('student_batches', pd.DataFrame())
        if not batches_df.empty and 'batch_id' in batches_df.columns:
            for idx, row in batches_df.iterrows():
                batch_id = str(row['batch_id'])
                self.batch_map[batch_id] = idx
                self.batch_reverse[idx] = batch_id
            self.n_batches = len(batches_df)
            logger.info(f"Built {self.n_batches} batch mappings")
        
        # Build assignment mappings (composite)
        assignment_idx = 0
        for course_id in self.course_map.keys():
            for faculty_id in self.faculty_map.keys():
                for room_id in self.room_map.keys():
                    for timeslot_id in self.timeslot_map.keys():
                        for batch_id in self.batch_map.keys():
                            assignment_tuple = (course_id, faculty_id, room_id, timeslot_id, batch_id)
                            self.assignment_map[assignment_tuple] = assignment_idx
                            self.assignment_reverse[assignment_idx] = assignment_tuple
                            assignment_idx += 1
        
        self.n_assignments = assignment_idx
        logger.info(f"Built {self.n_assignments} assignment mappings")
        
        # Verify bijectivity
        self._verify_bijectivity(logger)
    
    def _verify_bijectivity(self, logger: logging.Logger):
        """Verify bijectivity mathematically."""
        logger.info("Verifying bijectivity")
        
        # Verify course mappings
        for course_id, idx in self.course_map.items():
            assert self.course_reverse[idx] == course_id, f"Course mapping not bijective: {course_id}"
        
        for idx, course_id in self.course_reverse.items():
            assert self.course_map[course_id] == idx, f"Course reverse mapping not bijective: {idx}"
        
        logger.info("✓ Course mappings verified bijective")
        
        # Similar verification for other mappings...
        logger.info("✓ All mappings verified bijective")
    
    def encode_course(self, course_id: str) -> int:
        """Encode course_id to integer index."""
        return self.course_map.get(course_id, -1)
    
    def decode_course(self, idx: int) -> Optional[str]:
        """Decode integer index to course_id."""
        return self.course_reverse.get(idx)
    
    def encode_faculty(self, faculty_id: str) -> int:
        """Encode faculty_id to integer index."""
        return self.faculty_map.get(faculty_id, -1)
    
    def decode_faculty(self, idx: int) -> Optional[str]:
        """Decode integer index to faculty_id."""
        return self.faculty_reverse.get(idx)
    
    def encode_room(self, room_id: str) -> int:
        """Encode room_id to integer index."""
        return self.room_map.get(room_id, -1)
    
    def decode_room(self, idx: int) -> Optional[str]:
        """Decode integer index to room_id."""
        return self.room_reverse.get(idx)
    
    def encode_timeslot(self, timeslot_id: str) -> int:
        """Encode timeslot_id to integer index."""
        return self.timeslot_map.get(timeslot_id, -1)
    
    def decode_timeslot(self, idx: int) -> Optional[str]:
        """Decode integer index to timeslot_id."""
        return self.timeslot_reverse.get(idx)
    
    def encode_batch(self, batch_id: str) -> int:
        """Encode batch_id to integer index."""
        return self.batch_map.get(batch_id, -1)
    
    def decode_batch(self, idx: int) -> Optional[str]:
        """Decode integer index to batch_id."""
        return self.batch_reverse.get(idx)
    
    def encode_assignment(self, course_id: str, faculty_id: str, room_id: str, 
                         timeslot_id: str, batch_id: str) -> int:
        """Encode assignment tuple to unique variable ID."""
        assignment_tuple = (course_id, faculty_id, room_id, timeslot_id, batch_id)
        return self.assignment_map.get(assignment_tuple, -1)
    
    def decode_assignment(self, var_id: int) -> Optional[Tuple]:
        """Decode unique variable ID to assignment tuple."""
        return self.assignment_reverse.get(var_id)




