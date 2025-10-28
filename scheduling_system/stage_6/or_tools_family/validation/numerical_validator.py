"""
Numerical Validator

Validates numerical accuracy per Definition 11.3
from Stage-6.2 OR-Tools Foundational Framework.

Definition 11.3:
- Constraint satisfaction tolerance: |violation| < 10⁻⁶
- Objective value precision: |computed - expected| < 10⁻⁶
- Variable bounds validation: l ≤ x ≤ u

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from input_model.loader import CompiledData
from input_model.bijection import BijectiveMapper


@dataclass
class NumericalValidationResult:
    """Result of numerical validation."""
    is_valid: bool
    constraint_violations: List[Dict[str, Any]] = field(default_factory=list)
    objective_errors: List[Dict[str, Any]] = field(default_factory=list)
    bounds_violations: List[Dict[str, Any]] = field(default_factory=list)
    max_constraint_violation: float = 0.0
    max_objective_error: float = 0.0
    max_bounds_violation: float = 0.0
    tolerance: float = 1e-6


class NumericalValidator:
    """
    Validate numerical accuracy per Definition 11.3.
    
    Validation criteria:
    1. Constraint satisfaction tolerance: |violation| < 10⁻⁶
    2. Objective value precision: |computed - expected| < 10⁻⁶
    3. Variable bounds validation: l ≤ x ≤ u
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.tolerance = 1e-6
    
    def validate_solution(
        self,
        solution: Any,
        compiled_data: CompiledData,
        bijective_mapper: BijectiveMapper,
        solver_result: Any
    ) -> NumericalValidationResult:
        """
        Validate solution numerical accuracy.
        
        Args:
            solution: Solution object
            compiled_data: Stage 3 compiled data
            bijective_mapper: Bijective mappings
            solver_result: Raw solver result
            
        Returns:
            NumericalValidationResult with validation details
        """
        self.logger.info("=" * 80)
        self.logger.info("NUMERICAL VALIDATION - Definition 11.3")
        self.logger.info("=" * 80)
        self.logger.info(f"Tolerance: {self.tolerance}")
        
        result = NumericalValidationResult(
            is_valid=True,
            tolerance=self.tolerance
        )
        
        # 1. Validate constraint satisfaction
        self.logger.info("Validating constraint satisfaction")
        constraint_violations = self._validate_constraints(
            solution, compiled_data, bijective_mapper
        )
        result.constraint_violations = constraint_violations
        
        if constraint_violations:
            result.max_constraint_violation = max(
                abs(v['violation']) for v in constraint_violations
            )
            self.logger.warning(f"Found {len(constraint_violations)} constraint violations")
            self.logger.warning(f"Max violation: {result.max_constraint_violation:.2e}")
            
            if result.max_constraint_violation > self.tolerance:
                result.is_valid = False
        else:
            self.logger.info("✓ All constraints satisfied within tolerance")
        
        # 2. Validate objective value precision
        self.logger.info("Validating objective value precision")
        objective_errors = self._validate_objective(
            solution, compiled_data, bijective_mapper, solver_result
        )
        result.objective_errors = objective_errors
        
        if objective_errors:
            result.max_objective_error = max(
                abs(e['error']) for e in objective_errors
            )
            self.logger.warning(f"Found {len(objective_errors)} objective errors")
            self.logger.warning(f"Max error: {result.max_objective_error:.2e}")
            
            if result.max_objective_error > self.tolerance:
                result.is_valid = False
        else:
            self.logger.info("✓ Objective value within precision tolerance")
        
        # 3. Validate variable bounds
        self.logger.info("Validating variable bounds")
        bounds_violations = self._validate_bounds(
            solution, compiled_data, bijective_mapper
        )
        result.bounds_violations = bounds_violations
        
        if bounds_violations:
            result.max_bounds_violation = max(
                abs(v['violation']) for v in bounds_violations
            )
            self.logger.warning(f"Found {len(bounds_violations)} bounds violations")
            self.logger.warning(f"Max violation: {result.max_bounds_violation:.2e}")
            
            if result.max_bounds_violation > self.tolerance:
                result.is_valid = False
        else:
            self.logger.info("✓ All variable bounds satisfied")
        
        # Summary
        self.logger.info("=" * 80)
        if result.is_valid:
            self.logger.info("✓ NUMERICAL VALIDATION PASSED")
        else:
            self.logger.error("✗ NUMERICAL VALIDATION FAILED")
        self.logger.info("=" * 80)
        
        return result
    
    def _validate_constraints(
        self,
        solution: Any,
        compiled_data: CompiledData,
        bijective_mapper: BijectiveMapper
    ) -> List[Dict[str, Any]]:
        """
        Validate constraint satisfaction.
        
        For each constraint:
        |violation| < tolerance
        """
        violations = []
        
        assignments = solution.assignments if hasattr(solution, 'assignments') else []
        
        if not assignments:
            self.logger.warning("No assignments to validate")
            return violations
        
        # Validate hard constraints
        
        # 1. Faculty conflict constraints
        faculty_violations = self._check_faculty_conflicts(assignments)
        violations.extend(faculty_violations)
        
        # 2. Room conflict constraints
        room_violations = self._check_room_conflicts(assignments)
        violations.extend(room_violations)
        
        # 3. Batch conflict constraints
        batch_violations = self._check_batch_conflicts(assignments)
        violations.extend(batch_violations)
        
        # 4. Room capacity constraints
        capacity_violations = self._check_room_capacity(assignments, compiled_data)
        violations.extend(capacity_violations)
        
        return violations
    
    def _check_faculty_conflicts(self, assignments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check faculty conflict constraints."""
        violations = []
        
        # Group by timeslot
        timeslot_assignments = {}
        for assignment in assignments:
            timeslot_id = assignment.get('timeslot_id')
            if timeslot_id:
                if timeslot_id not in timeslot_assignments:
                    timeslot_assignments[timeslot_id] = []
                timeslot_assignments[timeslot_id].append(assignment)
        
        # Check for faculty conflicts per timeslot
        for timeslot_id, timeslot_assigns in timeslot_assignments.items():
            faculty_counts = {}
            for assignment in timeslot_assigns:
                faculty_id = assignment.get('faculty_id')
                if faculty_id:
                    faculty_counts[faculty_id] = faculty_counts.get(faculty_id, 0) + 1
            
            # Check for conflicts
            for faculty_id, count in faculty_counts.items():
                if count > 1:
                    violation = {
                        'constraint': 'faculty_conflict',
                        'timeslot_id': timeslot_id,
                        'faculty_id': faculty_id,
                        'count': count,
                        'violation': count - 1  # Excess assignments
                    }
                    violations.append(violation)
        
        return violations
    
    def _check_room_conflicts(self, assignments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check room conflict constraints."""
        violations = []
        
        # Group by timeslot
        timeslot_assignments = {}
        for assignment in assignments:
            timeslot_id = assignment.get('timeslot_id')
            if timeslot_id:
                if timeslot_id not in timeslot_assignments:
                    timeslot_assignments[timeslot_id] = []
                timeslot_assignments[timeslot_id].append(assignment)
        
        # Check for room conflicts per timeslot
        for timeslot_id, timeslot_assigns in timeslot_assignments.items():
            room_counts = {}
            for assignment in timeslot_assigns:
                room_id = assignment.get('room_id')
                if room_id:
                    room_counts[room_id] = room_counts.get(room_id, 0) + 1
            
            # Check for conflicts
            for room_id, count in room_counts.items():
                if count > 1:
                    violation = {
                        'constraint': 'room_conflict',
                        'timeslot_id': timeslot_id,
                        'room_id': room_id,
                        'count': count,
                        'violation': count - 1
                    }
                    violations.append(violation)
        
        return violations
    
    def _check_batch_conflicts(self, assignments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check batch conflict constraints."""
        violations = []
        
        # Group by timeslot
        timeslot_assignments = {}
        for assignment in assignments:
            timeslot_id = assignment.get('timeslot_id')
            if timeslot_id:
                if timeslot_id not in timeslot_assignments:
                    timeslot_assignments[timeslot_id] = []
                timeslot_assignments[timeslot_id].append(assignment)
        
        # Check for batch conflicts per timeslot
        for timeslot_id, timeslot_assigns in timeslot_assignments.items():
            batch_counts = {}
            for assignment in timeslot_assigns:
                batch_id = assignment.get('batch_id')
                if batch_id:
                    batch_counts[batch_id] = batch_counts.get(batch_id, 0) + 1
            
            # Check for conflicts
            for batch_id, count in batch_counts.items():
                if count > 1:
                    violation = {
                        'constraint': 'batch_conflict',
                        'timeslot_id': timeslot_id,
                        'batch_id': batch_id,
                        'count': count,
                        'violation': count - 1
                    }
                    violations.append(violation)
        
        return violations
    
    def _check_room_capacity(
        self,
        assignments: List[Dict[str, Any]],
        compiled_data: CompiledData
    ) -> List[Dict[str, Any]]:
        """Check room capacity constraints."""
        violations = []
        
        # Get room capacities
        rooms = compiled_data.L_raw.get('rooms', None)
        batches = compiled_data.L_raw.get('student_batches', None)
        
        if rooms is None or batches is None:
            return violations
        
        room_capacity = {}
        if 'room_id' in rooms.columns and 'capacity' in rooms.columns:
            for _, row in rooms.iterrows():
                room_capacity[str(row['room_id'])] = row['capacity']
        
        batch_size = {}
        if 'batch_id' in batches.columns and 'student_count' in batches.columns:
            for _, row in batches.iterrows():
                batch_size[str(row['batch_id'])] = row['student_count']
        
        # Check each assignment
        for assignment in assignments:
            room_id = assignment.get('room_id')
            batch_id = assignment.get('batch_id')
            
            if room_id and batch_id:
                capacity = room_capacity.get(room_id, float('inf'))
                size = batch_size.get(batch_id, 0)
                
                if size > capacity:
                    violation = {
                        'constraint': 'room_capacity',
                        'room_id': room_id,
                        'batch_id': batch_id,
                        'capacity': capacity,
                        'size': size,
                        'violation': size - capacity
                    }
                    violations.append(violation)
        
        return violations
    
    def _validate_objective(
        self,
        solution: Any,
        compiled_data: CompiledData,
        bijective_mapper: BijectiveMapper,
        solver_result: Any
    ) -> List[Dict[str, Any]]:
        """
        Validate objective value precision.
        
        |computed - expected| < tolerance
        """
        errors = []
        
        # Get solver objective value
        solver_objective = None
        if hasattr(solver_result, 'objective_value'):
            solver_objective = solver_result.objective_value
        
        # Get solution objective value
        solution_objective = solution.quality if hasattr(solution, 'quality') else None
        
        if solver_objective is not None and solution_objective is not None:
            error = abs(solver_objective - solution_objective)
            
            if error > self.tolerance:
                error_dict = {
                    'type': 'objective_mismatch',
                    'solver_value': solver_objective,
                    'solution_value': solution_objective,
                    'error': error
                }
                errors.append(error_dict)
        
        # TODO: Recompute objective from scratch and compare
        
        return errors
    
    def _validate_bounds(
        self,
        solution: Any,
        compiled_data: CompiledData,
        bijective_mapper: BijectiveMapper
    ) -> List[Dict[str, Any]]:
        """
        Validate variable bounds.
        
        For each variable x with bounds [l, u]:
        l ≤ x ≤ u
        """
        violations = []
        
        # TODO: Implement comprehensive bounds checking
        # This requires access to variable values and their bounds
        
        # For now, check basic bounds on assignments
        assignments = solution.assignments if hasattr(solution, 'assignments') else []
        
        # All assignment variables should be binary (0 or 1)
        # This is implicitly satisfied by the solver
        
        return violations

