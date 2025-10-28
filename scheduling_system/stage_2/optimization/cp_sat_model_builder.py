"""
OR-Tools CP-SAT Model Builder for Stage-2 Batching
Implements Definition 1.1-1.2 and Constraints from OR-Tools Bridge Foundation
"""

from ortools.sat.python import cp_model
import numpy as np
from typing import Dict, List, Tuple


class CPSATBatchingModel:
    """
    CP-SAT Batching Model Builder
    
    Implements:
    - Definition 1.1: BatchingCSP = (X_batch, D_batch, C_batch, F_multi)
    - Definition 1.2: Decision Variable Structure
    - Definition 3.1-3.3: Hard Constraints
    - Definition 4.1-4.2: Soft Constraints
    """
    
    def __init__(
        self,
        students: List[Dict],
        courses: List[Dict],
        rooms: List[Dict],
        parameters: Dict
    ):
        """
        Initialize CP-SAT batching model.
        
        Args:
            students: List of student records
            courses: List of course records
            rooms: List of room records
            parameters: Foundation parameters dictionary
        """
        self.model = cp_model.CpModel()
        self.students = students
        self.courses = courses
        self.rooms = rooms
        self.parameters = parameters
        
        self.n = len(students)
        self.m = self._estimate_batch_count()
        
        # Decision variables (will be built by build_decision_variables)
        self.x = {}  # x[i,j] ∈ {0,1}
        self.batch_size = []
        self.homogeneity = []
        self.dominant_shift = []

        # Enrollment map for coherence calculation (student -> set(course_id))
        self.student_index_to_courses: Dict[int, set] = {}

        # Course universe for batch course-set modeling
        self.course_ids: List[str] = [str(c.get('course_id')) for c in self.courses] if self.courses else []
        self.course_index_by_id: Dict[str, int] = {cid: idx for idx, cid in enumerate(self.course_ids)}
        # has_course[i][c_idx] = 1 if student i enrolled in course c
        self.has_course: List[List[int]] = []
        
        # Soft constraint penalty terms
        self.soft_penalty_terms = []
        
        # Metadata
        self.variable_count = 0
        self.constraint_count = 0
    
    def _estimate_batch_count(self) -> int:
        """
        Estimate optimal number of batches.
        
        Uses target batch size of 45 students per batch.
        """
        target_batch_size = 45
        return max(1, (self.n + target_batch_size - 1) // target_batch_size)
    
    def build_decision_variables(self) -> None:
        """
        Definition 1.2: Build Decision Variables
        
        x[i,j] ∈ {0,1} for all students i and batches j
        """
        # Primary decision variables: x[i,j] = 1 if student i in batch j
        for i in range(self.n):
            for j in range(self.m):
                self.x[i, j] = self.model.NewBoolVar(f'x_{i}_{j}')
                self.variable_count += 1
        
        # Auxiliary variables: batch sizes
        self.batch_size = [
            self.model.NewIntVar(
                self.parameters['min_batch_size'],
                self.parameters['max_batch_size'],
                f'batch_size_{j}'
            )
            for j in range(self.m)
        ]
        self.variable_count += self.m
        
        # Auxiliary variables: homogeneity scores (scaled to integers)
        self.homogeneity = [
            self.model.NewIntVar(0, 10000, f'homogeneity_{j}')
            for j in range(self.m)
        ]
        self.variable_count += self.m
        
        # Auxiliary variables: dominant shift for each batch
        max_shifts = len(set(s.get('preferred_shift') for s in self.students if s.get('preferred_shift')))
        if max_shifts == 0:
            max_shifts = 1
        
        self.dominant_shift = [
            self.model.NewIntVar(1, max_shifts, f'dominant_shift_{j}')
            for j in range(self.m)
        ]
        self.variable_count += self.m

        # Precompute enrollment sets for coherence
        for i, s in enumerate(self.students):
            enrolled = s.get('enrolled_courses', [])
            self.student_index_to_courses[i] = set(enrolled) if isinstance(enrolled, list) else set()

        # Build has_course matrix
        self.has_course = []
        for i in range(self.n):
            row = [0] * len(self.course_ids)
            for cid in self.student_index_to_courses[i]:
                c_str = str(cid)
                if c_str in self.course_index_by_id:
                    row[self.course_index_by_id[c_str]] = 1
            self.has_course.append(row)
    
    def add_assignment_constraints(self) -> None:
        """
        Definition 3.1: Assignment Constraint
        
        Each student assigned to exactly one batch: Σj xij = 1
        """
        for i in range(self.n):
            self.model.Add(sum(self.x[i, j] for j in range(self.m)) == 1)
            self.constraint_count += 1
    
    def add_capacity_constraints(self) -> None:
        """
        Definition 3.2: Capacity Constraints
        
        ℓj ≤ Σi xij ≤ uj for all batches j
        """
        for j in range(self.m):
            students_in_batch = sum(self.x[i, j] for i in range(self.n))
            
            # Lower bound
            self.model.Add(students_in_batch >= self.parameters['min_batch_size'])
            self.constraint_count += 1
            
            # Upper bound
            self.model.Add(students_in_batch <= self.parameters['max_batch_size'])
            self.constraint_count += 1
            
            # Link to batch_size variable
            self.model.Add(self.batch_size[j] == students_in_batch)
            self.constraint_count += 1
    
    def add_coherence_constraints(self, similarity_matrix: np.ndarray) -> None:
        """
        Definition 3.3: Course Coherence Constraint
        
        Students in same batch must share ≥75% course overlap relative to
        the batch course set Cbatch_j.
        """
        theta_num = int(self.parameters.get('coherence_threshold', 0.75) * 100)  # numerator (percent)

        num_courses = len(self.course_ids)
        if num_courses == 0:
            return

        # y[c,j] = 1 if course c is present in batch j (any assigned student has it)
        self.y_course_in_batch = [
            [self.model.NewBoolVar(f'y_course_{c}_{j}') for j in range(self.m)]
            for c in range(num_courses)
        ]
        self.variable_count += num_courses * self.m

        # Link y[c,j] with x[i,j]: if any student i with has_course[i][c] == 1 is assigned, y[c,j] must be 1
        for c in range(num_courses):
            for j in range(self.m):
                # y[c,j] >= x[i,j] for all i with has_course[i][c] == 1
                interested = [i for i in range(self.n) if self.has_course[i][c] == 1]
                if interested:
                    for i in interested:
                        self.model.Add(self.y_course_in_batch[c][j] >= self.x[i, j])
                        self.constraint_count += 1
                else:
                    # If no student has this course, ensure y[c,j] == 0
                    self.model.Add(self.y_course_in_batch[c][j] == 0)
                    self.constraint_count += 1

        # |Cbatch_j| size variable
        self.cbatch_size = [self.model.NewIntVar(0, num_courses, f'cbatch_size_{j}') for j in range(self.m)]
        self.variable_count += self.m
        for j in range(self.m):
            self.model.Add(self.cbatch_size[j] == sum(self.y_course_in_batch[c][j] for c in range(num_courses)))
            self.constraint_count += 1

        # For each student i and batch j, count intersection size inter[i,j] = sum over c of z[i,c,j]
        # where z[i,c,j] = 1 only if student i has course c and y[c,j] == 1
        for j in range(self.m):
            for i in range(self.n):
                # Create z variables only for courses that student i has
                z_vars = []
                for c in range(num_courses):
                    if self.has_course[i][c] == 1:
                        z = self.model.NewBoolVar(f'z_inter_{i}_{c}_{j}')
                        # z <= y[c,j]
                        self.model.Add(z <= self.y_course_in_batch[c][j])
                        # If student does not have course c, would be 0; here it's ensured by selection
                        z_vars.append(z)
                        self.variable_count += 1
                inter_ij = self.model.NewIntVar(0, num_courses, f'inter_{i}_{j}')
                self.variable_count += 1
                if z_vars:
                    self.model.Add(inter_ij == sum(z_vars))
                else:
                    self.model.Add(inter_ij == 0)
                self.constraint_count += 1

                # Enforce inter_ij * 100 >= theta_num * |Cbatch_j| if x[i,j] == 1
                # Avoid division by cross-multiplication
                self.model.Add(inter_ij * 100 >= theta_num * self.cbatch_size[j]).OnlyEnforceIf(self.x[i, j])
                self.constraint_count += 1
    
    def _compute_batch_coherence_score(
        self,
        student_idx: int,
        batch_idx: int,
        similarity_matrix: np.ndarray
    ) -> int:
        # Deprecated by direct constraint with similarity threshold; retained for API compatibility.
        return int(self.parameters.get('coherence_threshold', 0.75) * 100)
    
    def add_shift_preference_constraints(self) -> None:
        """
        Definition 4.1: Shift Preference Penalties

        Minimize conflicts with preferred time shifts by penalizing
        total assigned minus the dominant shift count per batch.
        """
        shift_penalty_weight = int(self.parameters.get('shift_preference_penalty', 2.0) * 100)

        # Collect distinct shift values from students (map them to compact ints)
        shift_values = []
        for s in self.students:
            v = s.get('preferred_shift')
            if v is not None and v not in shift_values:
                shift_values.append(v)
        if not shift_values:
            return

        # For each batch, compute counts per shift and penalize non-dominant assignments
        for j in range(self.m):
            count_vars = []
            for sv in shift_values:
                # count of students with shift == sv assigned to batch j
                count_sv_j = self.model.NewIntVar(0, self.n, f'shift_count_{j}_{sv}')
                # Build sum of x[i,j] over students whose shift equals sv
                terms = []
                for i in range(self.n):
                    if self.students[i].get('preferred_shift') == sv:
                        terms.append(self.x[i, j])
                if terms:
                    self.model.Add(count_sv_j == sum(terms))
                else:
                    self.model.Add(count_sv_j == 0)
                count_vars.append(count_sv_j)

            # max count among shifts
            max_shift_count = self.model.NewIntVar(0, self.n, f'max_shift_count_{j}')
            self.model.AddMaxEquality(max_shift_count, count_vars)

            # shift violations = total assigned - dominant shift count
            shift_violations = self.model.NewIntVar(0, self.n, f'shift_violations_{j}')
            self.model.Add(shift_violations == self.batch_size[j] - max_shift_count)

            self.soft_penalty_terms.append(shift_penalty_weight * shift_violations)
    
    def add_language_compatibility_constraints(self, language_preferences: Dict = None) -> None:
        """
        Definition 4.2: Language Compatibility
        
        Promote language homogeneity within batches by penalizing
        non-dominant language assignments.
        """
        language_penalty_weight = int(self.parameters.get('language_mismatch_penalty', 1.5) * 100)

        # Collect distinct primary languages
        langs = []
        for s in self.students:
            li = s.get('primary_instruction_language') or s.get('language')
            if li and li not in langs:
                langs.append(li)
        if not langs:
            return

        for j in range(self.m):
            count_vars = []
            for lv in langs:
                count_lv_j = self.model.NewIntVar(0, self.n, f'lang_count_{j}_{lv}')
                terms = []
                for i in range(self.n):
                    li = self.students[i].get('primary_instruction_language') or self.students[i].get('language')
                    if li == lv:
                        terms.append(self.x[i, j])
                if terms:
                    self.model.Add(count_lv_j == sum(terms))
                else:
                    self.model.Add(count_lv_j == 0)
                count_vars.append(count_lv_j)

            max_lang_count = self.model.NewIntVar(0, self.n, f'max_lang_count_{j}')
            self.model.AddMaxEquality(max_lang_count, count_vars)
            lang_violations = self.model.NewIntVar(0, self.n, f'lang_violations_{j}')
            self.model.Add(lang_violations == self.batch_size[j] - max_lang_count)
            self.soft_penalty_terms.append(language_penalty_weight * lang_violations)
    
    def build_combined_objective(self, similarity_matrix: np.ndarray) -> None:
        """
        Definition 2.2: Weighted Sum Scalarization
        
        F_total = w1·f1 + w2·(-f2) + w3·f3
        """
        from stage_2.optimization.objective_functions import ObjectiveFunctionBuilder
        
        obj_builder = ObjectiveFunctionBuilder(self.model, self)
        
        # Build f1: Batch size optimization
        f1 = obj_builder.build_f1_objective(self._get_target_sizes())
        
        # Build f2: Academic homogeneity (maximize → minimize negative)
        f2 = obj_builder.build_f2_objective(similarity_matrix)
        
        # Build f3: Resource utilization balance
        # Resource demand per batch as integer: number of students in batch as proxy
        # Create IntVar demands equal to batch_size[j]
        self.total_student_demand = self.n
        batch_demands = self.batch_size
        f3 = obj_builder.build_f3_from_intvars(batch_demands)
        
        # Weighted combination
        w1 = int(self.parameters['size_weight'] * 1000)
        w2 = int(self.parameters['homogeneity_weight'] * 1000)
        w3 = int(self.parameters['balance_weight'] * 1000)
        
        # Add soft constraint penalties
        soft_penalty = sum(self.soft_penalty_terms) if self.soft_penalty_terms else 0
        
        total_objective = w1 * f1 + w2 * f2 + w3 * f3 + soft_penalty
        
        self.model.Minimize(total_objective)
    
    def _get_target_sizes(self) -> List[int]:
        """Get target batch sizes."""
        target_size = self.n // self.m
        return [target_size] * self.m
    
    def _get_resource_demands(self) -> List[float]:
        """Get resource demands for each batch."""
        # Simplified - returns uniform demands
        return [1.0] * self.m
    
    def get_variable_count(self) -> int:
        """Get total number of variables in model."""
        return self.variable_count
    
    def get_constraint_count(self) -> int:
        """Get total number of constraints in model."""
        return self.constraint_count
    
    def get_canonical_ordering(self) -> List[str]:
        """Get canonical ordering of students."""
        return [s['student_id'] for s in self.students]

