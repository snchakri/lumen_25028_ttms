"""
Core Data Structures for Stage 7 Validation
==========================================

Defines all data structures used in validation process.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set
from pathlib import Path
import pandas as pd


@dataclass
class Assignment:
    """Single schedule assignment (c, f, r, t, b)."""
    course_id: str
    faculty_id: str
    room_id: str
    timeslot_id: str
    batch_id: str
    assignment_id: Optional[str] = None
    day: Optional[str] = None
    time: Optional[str] = None
    duration: Optional[int] = None
    
    def to_tuple(self) -> tuple:
        """Convert to tuple (c, f, r, t, b)."""
        return (self.course_id, self.faculty_id, self.room_id, self.timeslot_id, self.batch_id)


@dataclass
class Schedule:
    """
    Complete schedule S = (A, Q) per Definition 2.1.
    
    A: Assignment set
    Q: Quality function
    """
    assignments: List[Assignment] = field(default_factory=list)
    objective_value: Optional[float] = None
    solver_used: str = ""
    solve_time: float = 0.0
    
    def get_assignment_set(self) -> Set[tuple]:
        """Get set of assignments A."""
        return {a.to_tuple() for a in self.assignments}
    
    def get_courses_scheduled(self) -> Set[str]:
        """Get set of scheduled courses."""
        return {a.course_id for a in self.assignments}
    
    def get_assignments_by_faculty(self, faculty_id: str) -> List[Assignment]:
        """Get all assignments for a faculty member."""
        return [a for a in self.assignments if a.faculty_id == faculty_id]
    
    def get_assignments_by_room(self, room_id: str) -> List[Assignment]:
        """Get all assignments for a room."""
        return [a for a in self.assignments if a.room_id == room_id]
    
    def get_assignments_by_batch(self, batch_id: str) -> List[Assignment]:
        """Get all assignments for a batch."""
        return [a for a in self.assignments if a.batch_id == batch_id]
    
    def get_assignments_by_timeslot(self, timeslot_id: str) -> List[Assignment]:
        """Get all assignments for a timeslot."""
        return [a for a in self.assignments if a.timeslot_id == timeslot_id]


@dataclass
class Stage3Data:
    """Data from Stage 3 compilation."""
    # Core entities
    institutions: pd.DataFrame
    departments: pd.DataFrame
    programs: pd.DataFrame
    courses: pd.DataFrame
    shifts: pd.DataFrame
    time_slots: pd.DataFrame
    faculty: pd.DataFrame
    rooms: pd.DataFrame
    batches: pd.DataFrame
    
    # Relationships
    faculty_course_competency: pd.DataFrame
    batch_course_enrollment: pd.DataFrame
    course_prerequisites: Optional[pd.DataFrame] = None
    room_department_access: Optional[pd.DataFrame] = None
    
    # Dynamic parameters
    dynamic_constraints: Optional[pd.DataFrame] = None
    dynamic_parameters: Optional[pd.DataFrame] = None
    
    def get_all_courses(self) -> Set[str]:
        """Get set of all course IDs."""
        return set(self.courses['course_id'].values)
    
    def get_required_courses(self) -> Set[str]:
        """Get set of required courses (non-elective)."""
        if 'course_type' in self.courses.columns:
            return set(self.courses[
                self.courses['course_type'].isin(['CORE', 'COMPULSORY'])
            ]['course_id'].values)
        return self.get_all_courses()
    
    def get_faculty_for_course(self, course_id: str) -> Set[str]:
        """Get competent faculty for a course."""
        competent = self.faculty_course_competency[
            (self.faculty_course_competency['course_id'] == course_id) &
            (self.faculty_course_competency['competency_level'] >= 5)  # Minimum competency
        ]
        return set(competent['faculty_id'].values)
    
    def get_batches_for_course(self, course_id: str) -> Set[str]:
        """Get batches enrolled in a course."""
        enrolled = self.batch_course_enrollment[
            self.batch_course_enrollment['course_id'] == course_id
        ]
        return set(enrolled['batch_id'].values)
    
    def get_room_capacity(self, room_id: str) -> int:
        """Get room capacity."""
        room = self.rooms[self.rooms['room_id'] == room_id]
        if len(room) > 0:
            return int(room.iloc[0]['capacity'])
        return 0
    
    def get_batch_size(self, batch_id: str) -> int:
        """Get batch size."""
        batch = self.batches[self.batches['batch_id'] == batch_id]
        if len(batch) > 0:
            return int(batch.iloc[0]['student_count'])
        return 0
    
    def get_course_hours(self, course_id: str) -> int:
        """Get weekly hours for a course."""
        course = self.courses[self.courses['course_id'] == course_id]
        if len(course) > 0:
            theory_hrs = course.iloc[0].get('theory_hours', 0) or 0
            practical_hrs = course.iloc[0].get('practical_hours', 0) or 0
            return int(theory_hrs + practical_hrs)
        return 0
    
    def get_faculty_preferences(self, faculty_id: str) -> Dict[str, float]:
        """Get faculty preferences for courses."""
        prefs = self.faculty_course_competency[
            self.faculty_course_competency['faculty_id'] == faculty_id
        ]
        return dict(zip(prefs['course_id'], prefs.get('preference_score', prefs['competency_level'])))
    
    def get_prerequisite_pairs(self) -> List[tuple]:
        """Get prerequisite pairs (c1, c2) where c1 is prerequisite for c2."""
        if self.course_prerequisites is None or len(self.course_prerequisites) == 0:
            return []
        
        pairs = []
        for _, row in self.course_prerequisites.iterrows():
            prereq = row['prerequisite_course_id']
            course = row['course_id']
            pairs.append((prereq, course))
        
        return pairs


@dataclass
class ThresholdResult:
    """Result of threshold validation."""
    threshold_id: str
    value: float
    lower_bound: float
    upper_bound: float
    target: Optional[float]
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    computation_time_ms: float = 0.0
    
    def __post_init__(self):
        """Validate result."""
        if self.value < 0 or self.value > 1:
            # Allow values slightly outside [0,1] due to numerical precision
            if not (-1e-6 <= self.value <= 1.0 + 1e-6):
                raise ValueError(f"Threshold value {self.value} outside valid range [0, 1]")


@dataclass
class ValidationResult:
    """Complete validation result for all thresholds."""
    session_id: str
    schedule_file: str
    timestamp: str
    
    # Individual threshold results
    threshold_results: Dict[str, ThresholdResult] = field(default_factory=dict)
    
    # Global quality score (Definition 2.1)
    global_quality_score: Optional[float] = None
    
    # Validation status
    all_passed: bool = False
    critical_failures: List[str] = field(default_factory=list)
    
    # Performance metrics
    total_validation_time_ms: float = 0.0
    
    def add_threshold_result(self, result: ThresholdResult):
        """Add threshold result."""
        self.threshold_results[result.threshold_id] = result
        if not result.passed:
            self.critical_failures.append(result.threshold_id)
    
    def compute_global_quality(self, weights: Dict[str, float]):
        """
        Compute global quality score per Definition 2.1.
        
        Q_global(S) = Σ wᵢ · φᵢ(S)
        """
        total_quality = 0.0
        for threshold_id, weight in weights.items():
            if threshold_id in self.threshold_results:
                total_quality += weight * self.threshold_results[threshold_id].value
        
        self.global_quality_score = total_quality
        
        # Check if all thresholds passed
        self.all_passed = len(self.critical_failures) == 0
        
        return total_quality
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary and make JSON safe."""
        raw = {
            'session_id': self.session_id,
            'schedule_file': self.schedule_file,
            'timestamp': self.timestamp,
            'global_quality_score': self.global_quality_score,
            'all_passed': self.all_passed,
            'critical_failures': self.critical_failures,
            'total_validation_time_ms': self.total_validation_time_ms,
            'threshold_results': {
                tid: {
                    'value': res.value,
                    'bounds': {'lower': res.lower_bound, 'upper': res.upper_bound},
                    'target': res.target,
                    'passed': res.passed,
                    'computation_time_ms': res.computation_time_ms,
                    'details': res.details
                }
                for tid, res in self.threshold_results.items()
            }
        }
        from scheduling_engine_localized.stage_7.logging_system.logger import json_safe
        return json_safe(raw)
