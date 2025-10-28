"""
PyGMO Problem Formulation for Educational Scheduling

Implements the multi-objective scheduling problem for PyGMO optimization.

Theoretical Foundation:
- PyGMO SOLVER FAMILY - Foundational Framework
- Section 2.2: Timetabling Problem Mapping
- Section 8.1: Problem-Specific Adaptations
- Equations (1)-(11): Multi-objective formulation

This module implements:
- 5-objective fitness function
- Mixed discrete/continuous variables
- Constraint handling (hard and soft)
- Solution encoding/decoding
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class ProblemDimensions:
    """Problem dimensions for variable structure"""
    n_courses: int
    n_faculty: int
    n_rooms: int
    n_timeslots: int
    n_batches: int
    
    # Variable counts
    discrete_count: int
    continuous_count: int
    total_dimensions: int
    
    def __post_init__(self):
        """Validate dimensions"""
        assert self.n_courses > 0, "Must have at least one course"
        assert self.n_faculty > 0, "Must have at least one faculty"
        assert self.n_rooms > 0, "Must have at least one room"
        assert self.n_timeslots > 0, "Must have at least one timeslot"
        assert self.n_batches > 0, "Must have at least one batch"
        assert self.total_dimensions == self.discrete_count + self.continuous_count


class SchedulingProblem:
    """
    PyGMO-compatible scheduling problem.
    
    Implements the UDP (User-Defined Problem) interface for PyGMO.
    
    Decision Variables:
    - Discrete: x[c,f,r,t,b] ∈ {0,1} for assignments
    - Continuous: preference weights, satisfaction scores ∈ [0,1]
    
    Objectives (minimize all):
    - f₁: Conflict penalty (hard constraint violations)
    - f₂: -Resource utilization (maximize → minimize negative)
    - f₃: -Preference satisfaction (maximize → minimize negative)
    - f₄: Workload imbalance (variance minimization)
    - f₅: Schedule fragmentation (gap minimization)
    
    Theoretical Foundation: Equations (7)-(11) from Section 8.1
    """
    
    def __init__(
        self,
        compiled_data: Any,
        config: Optional[Any] = None,
        logger: Optional[Any] = None
    ):
        """
        Initialize scheduling problem.
        
        Args:
            compiled_data: CompiledData from InputLoader
            config: Optional PyGMOConfig instance
            logger: Optional StructuredLogger instance
        """
        self.compiled_data = compiled_data
        self.config = config
        self.logger = logger
        
        # Extract core entities - try different entity name variants
        self.courses = self._get_entity('courses', 'course')
        self.faculty = self._get_entity('faculty')
        self.rooms = self._get_entity('rooms', 'room')
        self.timeslots = self._get_entity('time_slots', 'timeslots', 'scheduling_sessions')
        self.batches = self._get_entity('batches', 'student_batches')
        self.students = self._get_entity('students', 'student')
        self.competency = self._get_entity('faculty_course_competency', 'competency')
        self.constraints_df = self._get_entity('constraints', 'dynamic_constraints')
        
        # Log loaded entities
        if self.logger:
            self.logger.info(f"Loaded entities - courses: {len(self.courses) if self.courses is not None else 0}, "
                           f"faculty: {len(self.faculty) if self.faculty is not None else 0}, "
                           f"rooms: {len(self.rooms) if self.rooms is not None else 0}, "
                           f"timeslots: {len(self.timeslots) if self.timeslots is not None else 0}, "
                           f"batches: {len(self.batches) if self.batches is not None else 0}")
        
        # Validate required entities
        self._validate_entities()
        
        # Calculate problem dimensions
        self.dimensions = self._calculate_dimensions()
        
        # Variable bounds
        self.lower_bounds, self.upper_bounds = self._calculate_bounds()
        
        # Discrete variable indices
        self.discrete_indices = list(range(self.dimensions.discrete_count))
        
        # Precompute lookup structures for efficiency
        self._build_lookup_structures()
        
        if self.logger:
            self.logger.info(
                "Scheduling problem initialized",
                dimensions=self.dimensions.total_dimensions,
                discrete=self.dimensions.discrete_count,
                continuous=self.dimensions.continuous_count,
                courses=self.dimensions.n_courses,
                faculty=self.dimensions.n_faculty,
                rooms=self.dimensions.n_rooms,
                timeslots=self.dimensions.n_timeslots,
                batches=self.dimensions.n_batches
            )
    
    def _get_entity(self, *names):
        """Try multiple entity names and return first found"""
        for name in names:
            try:
                df = self.compiled_data.get_entity_dataframe(name)
                if df is not None and not df.empty:
                    return df
            except:
                continue
        # Return empty DataFrame if none found
        import pandas as pd
        return pd.DataFrame()
    
    def _validate_entities(self):
        """Validate required entities are present"""
        required = ['courses', 'faculty', 'rooms', 'timeslots', 'batches']
        missing = []
        
        for entity in required:
            df = getattr(self, entity, None)
            if df is None or df.empty:
                missing.append(entity)
        
        if missing:
            raise ValueError(f"Missing required entities: {missing}")
    
    def _calculate_dimensions(self) -> ProblemDimensions:
        """
        Calculate problem dimensions.
        
        Returns:
            ProblemDimensions instance
        """
        n_courses = len(self.courses)
        n_faculty = len(self.faculty)
        n_rooms = len(self.rooms)
        n_timeslots = len(self.timeslots)
        n_batches = len(self.batches)
        
        # Discrete variables: binary assignment variables
        # x[c,f,r,t,b] for each (course, faculty, room, timeslot, batch) combination
        discrete_count = n_courses * n_faculty * n_rooms * n_timeslots * n_batches
        
        # Continuous variables: preference weights and balance factors
        # - Faculty preferences: n_faculty
        # - Course weights: n_courses
        # - Balance factors: n_faculty
        continuous_count = n_faculty * 2 + n_courses
        
        total_dimensions = discrete_count + continuous_count
        
        return ProblemDimensions(
            n_courses=n_courses,
            n_faculty=n_faculty,
            n_rooms=n_rooms,
            n_timeslots=n_timeslots,
            n_batches=n_batches,
            discrete_count=discrete_count,
            continuous_count=continuous_count,
            total_dimensions=total_dimensions
        )
    
    def _calculate_bounds(self) -> Tuple[List[float], List[float]]:
        """
        Calculate variable bounds for PyGMO.
        
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        lower = []
        upper = []
        
        # Discrete binary variables (0 or 1)
        for _ in range(self.dimensions.discrete_count):
            lower.append(0.0)
            upper.append(1.0)
        
        # Continuous variables (0.0 to 1.0)
        for _ in range(self.dimensions.continuous_count):
            lower.append(0.0)
            upper.append(1.0)
        
        return lower, upper
    
    def _build_lookup_structures(self):
        """Build lookup structures for efficient fitness evaluation"""
        # Course ID to index mapping
        self.course_id_to_idx = {
            cid: idx for idx, cid in enumerate(self.courses.iloc[:, 0])
        }
        
        # Faculty ID to index mapping
        self.faculty_id_to_idx = {
            fid: idx for idx, fid in enumerate(self.faculty.iloc[:, 0])
        }
        
        # Room ID to index mapping
        self.room_id_to_idx = {
            rid: idx for idx, rid in enumerate(self.rooms.iloc[:, 0])
        }
        
        # Timeslot ID to index mapping
        self.timeslot_id_to_idx = {
            tid: idx for idx, tid in enumerate(self.timeslots.iloc[:, 0])
        }
        
        # Batch ID to index mapping
        self.batch_id_to_idx = {
            bid: idx for idx, bid in enumerate(self.batches.iloc[:, 0])
        }
        
        # Competency matrix (faculty x course)
        self.competency_matrix = self._build_competency_matrix()
        
        # Room capacity vector
        self.room_capacities = self._extract_room_capacities()
        
        # Batch sizes
        self.batch_sizes = self._extract_batch_sizes()
    
    def _build_competency_matrix(self) -> np.ndarray:
        """Build competency matrix for faculty-course assignments"""
        matrix = np.zeros((self.dimensions.n_faculty, self.dimensions.n_courses))
        
        if self.competency is not None and not self.competency.empty:
            for _, row in self.competency.iterrows():
                fid = row.get('faculty_id')
                cid = row.get('course_id')
                comp = row.get('competency_level', 5.0)
                
                if fid in self.faculty_id_to_idx and cid in self.course_id_to_idx:
                    f_idx = self.faculty_id_to_idx[fid]
                    c_idx = self.course_id_to_idx[cid]
                    matrix[f_idx, c_idx] = comp
        
        return matrix
    
    def _extract_room_capacities(self) -> np.ndarray:
        """Extract room capacities"""
        capacities = np.zeros(self.dimensions.n_rooms)
        
        for idx, row in self.rooms.iterrows():
            cap = row.get('capacity', 50)  # Default capacity
            capacities[idx] = cap
        
        return capacities
    
    def _extract_batch_sizes(self) -> np.ndarray:
        """Extract batch sizes"""
        sizes = np.zeros(self.dimensions.n_batches)
        
        for idx, row in self.batches.iterrows():
            size = row.get('student_count', 30)  # Default size
            sizes[idx] = size
        
        return sizes
    
    # ========================================================================
    # PyGMO UDP Interface Methods
    # ========================================================================
    
    def fitness(self, x: List[float]) -> List[float]:
        """
        Evaluate fitness for decision vector x.
        
        Args:
            x: Decision vector [discrete_vars | continuous_vars]
        
        Returns:
            List of 5 objective values [f₁, f₂, f₃, f₄, f₅]
        
        Theoretical Foundation: Equations (7)-(11)
        """
        # Split decision vector
        discrete_vars = np.array(x[:self.dimensions.discrete_count])
        continuous_vars = np.array(x[self.dimensions.discrete_count:])
        
        # Decode assignments from discrete variables
        assignments = self._decode_assignments(discrete_vars)
        
        # Extract continuous parameters
        faculty_prefs = continuous_vars[:self.dimensions.n_faculty]
        course_weights = continuous_vars[
            self.dimensions.n_faculty:self.dimensions.n_faculty + self.dimensions.n_courses
        ]
        balance_factors = continuous_vars[-self.dimensions.n_faculty:]
        
        # Calculate 5 objectives
        f1_conflicts = self._calculate_conflicts(assignments)
        f2_utilization = self._calculate_utilization(assignments)
        f3_preferences = self._calculate_preference_satisfaction(
            assignments, faculty_prefs, course_weights
        )
        f4_balance = self._calculate_workload_balance(assignments, balance_factors)
        f5_compactness = self._calculate_schedule_compactness(assignments)
        
        return [f1_conflicts, f2_utilization, f3_preferences, f4_balance, f5_compactness]
    
    def get_bounds(self) -> Tuple[List[float], List[float]]:
        """
        Get variable bounds.
        
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        return self.lower_bounds, self.upper_bounds
    
    def get_nobj(self) -> int:
        """
        Get number of objectives.
        
        Returns:
            Number of objectives (5)
        """
        return 5
    
    def get_nix(self) -> int:
        """
        Get number of integer/discrete variables.
        
        Returns:
            Number of discrete variables
        """
        return self.dimensions.discrete_count
    
    def get_name(self) -> str:
        """Get problem name"""
        return "Educational_Scheduling_Problem"
    
    # ========================================================================
    # Assignment Decoding
    # ========================================================================
    
    def _decode_assignments(self, discrete_vars: np.ndarray) -> np.ndarray:
        """
        Decode discrete variables to assignment matrix.
        
        Args:
            discrete_vars: Binary decision variables
        
        Returns:
            Assignment matrix of shape (n_courses, n_faculty, n_rooms, n_timeslots, n_batches)
        """
        # Reshape flat vector to 5D assignment tensor
        shape = (
            self.dimensions.n_courses,
            self.dimensions.n_faculty,
            self.dimensions.n_rooms,
            self.dimensions.n_timeslots,
            self.dimensions.n_batches
        )
        
        # Round to binary (0 or 1)
        binary_vars = np.round(discrete_vars).astype(int)
        
        # Reshape
        assignments = binary_vars.reshape(shape)
        
        return assignments
    
    # ========================================================================
    # Objective Functions (Equations 7-11)
    # ========================================================================
    
    def _calculate_conflicts(self, assignments: np.ndarray) -> float:
        """
        Calculate conflict penalty (f₁).
        
        Equation (7): f₁(x) = Σᵢ wᵢ · conflictᵢ(x)
        
        Conflicts:
        - Faculty double-booking
        - Room double-booking
        - Batch double-booking
        - Unassigned courses
        - Incompetent assignments
        """
        penalty = 0.0
        
        # Faculty conflicts: faculty teaching multiple courses at same time
        for f in range(self.dimensions.n_faculty):
            for t in range(self.dimensions.n_timeslots):
                # Count assignments for this faculty at this time
                count = np.sum(assignments[:, f, :, t, :])
                if count > 1:
                    penalty += (count - 1) * 1000.0  # High penalty
        
        # Room conflicts: room hosting multiple classes at same time
        for r in range(self.dimensions.n_rooms):
            for t in range(self.dimensions.n_timeslots):
                count = np.sum(assignments[:, :, r, t, :])
                if count > 1:
                    penalty += (count - 1) * 1000.0
        
        # Batch conflicts: batch attending multiple classes at same time
        for b in range(self.dimensions.n_batches):
            for t in range(self.dimensions.n_timeslots):
                count = np.sum(assignments[:, :, :, t, b])
                if count > 1:
                    penalty += (count - 1) * 1000.0
        
        # Unassigned courses
        for c in range(self.dimensions.n_courses):
            count = np.sum(assignments[c, :, :, :, :])
            if count == 0:
                penalty += 500.0  # Course not assigned
            elif count > 1:
                penalty += (count - 1) * 100.0  # Course assigned multiple times
        
        # Incompetent assignments (faculty without competency)
        for c in range(self.dimensions.n_courses):
            for f in range(self.dimensions.n_faculty):
                if self.competency_matrix[f, c] < 5.0:  # Minimum competency threshold
                    # Penalty for assigning incompetent faculty
                    penalty += np.sum(assignments[c, f, :, :, :]) * 200.0
        
        return penalty
    
    def _calculate_utilization(self, assignments: np.ndarray) -> float:
        """
        Calculate resource underutilization (f₂).
        
        Equation (8): f₂(x) = Σⱼ (1 - utilizationⱼ(x))
        
        Minimize negative utilization (maximize utilization)
        """
        # Faculty utilization
        faculty_util = np.zeros(self.dimensions.n_faculty)
        for f in range(self.dimensions.n_faculty):
            assigned_hours = np.sum(assignments[:, f, :, :, :])
            max_hours = self.dimensions.n_timeslots * 0.5  # Assume 50% max load
            faculty_util[f] = assigned_hours / max_hours if max_hours > 0 else 0.0
        
        # Room utilization
        room_util = np.zeros(self.dimensions.n_rooms)
        for r in range(self.dimensions.n_rooms):
            assigned_slots = np.sum(assignments[:, :, r, :, :])
            total_slots = self.dimensions.n_timeslots
            room_util[r] = assigned_slots / total_slots if total_slots > 0 else 0.0
        
        # Calculate underutilization
        faculty_underutil = np.sum(1.0 - faculty_util)
        room_underutil = np.sum(1.0 - room_util)
        
        # Return negative (to minimize, which maximizes utilization)
        return -(faculty_underutil + room_underutil)
    
    def _calculate_preference_satisfaction(
        self,
        assignments: np.ndarray,
        faculty_prefs: np.ndarray,
        course_weights: np.ndarray
    ) -> float:
        """
        Calculate preference violation (f₃).
        
        Equation (9): f₃(x) = Σₖ penaltyₖ(x)
        
        Minimize negative satisfaction (maximize satisfaction)
        """
        satisfaction = 0.0
        
        # Faculty-course preferences
        for c in range(self.dimensions.n_courses):
            for f in range(self.dimensions.n_faculty):
                if np.sum(assignments[c, f, :, :, :]) > 0:
                    # Faculty assigned to course
                    pref_score = faculty_prefs[f] * course_weights[c]
                    satisfaction += pref_score
        
        # Normalize by number of assignments
        total_assignments = np.sum(assignments)
        if total_assignments > 0:
            satisfaction /= total_assignments
        
        # Return negative (to minimize, which maximizes satisfaction)
        return -satisfaction
    
    def _calculate_workload_balance(
        self,
        assignments: np.ndarray,
        balance_factors: np.ndarray
    ) -> float:
        """
        Calculate workload imbalance (f₄).
        
        Equation (10): f₄(x) = Var(workloads(x))
        
        Minimize variance in workload distribution
        """
        # Calculate workload for each faculty
        workloads = np.zeros(self.dimensions.n_faculty)
        for f in range(self.dimensions.n_faculty):
            workloads[f] = np.sum(assignments[:, f, :, :, :]) * balance_factors[f]
        
        # Calculate variance
        if len(workloads) > 1:
            variance = np.var(workloads)
        else:
            variance = 0.0
        
        return variance
    
    def _calculate_schedule_compactness(self, assignments: np.ndarray) -> float:
        """
        Calculate schedule fragmentation (f₅).
        
        Equation (11): f₅(x) = Σₗ gapsₗ(x)
        
        Minimize gaps in schedules
        """
        total_gaps = 0.0
        
        # Calculate gaps for each batch
        for b in range(self.dimensions.n_batches):
            # Get timeslots where batch has classes
            batch_schedule = np.sum(assignments[:, :, :, :, b], axis=(0, 1, 2))
            occupied_slots = np.where(batch_schedule > 0)[0]
            
            if len(occupied_slots) > 1:
                # Calculate gaps between classes
                min_slot = occupied_slots.min()
                max_slot = occupied_slots.max()
                span = max_slot - min_slot + 1
                actual_classes = len(occupied_slots)
                gaps = span - actual_classes
                total_gaps += gaps
        
        return total_gaps


