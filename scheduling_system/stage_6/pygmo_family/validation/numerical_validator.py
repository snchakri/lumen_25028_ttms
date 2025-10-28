"""
Numerical Validator Module

Validates numerical precision and constraint satisfaction.

Validations:
- Float64 precision ≤ 10⁻⁶
- No NaN/Inf values
- Constraint satisfaction
- Objective bounds
"""

import numpy as np
from typing import Dict, Any, List, Tuple

from ..config import PyGMOConfig
from ..logging_system.logger import StructuredLogger


class NumericalValidator:
    """
    Validates numerical precision and constraint satisfaction.
    """
    
    def __init__(self, config: PyGMOConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger
        self.precision_tolerance = 1e-6
        
        self.logger.info("NumericalValidator initialized successfully.")
    
    def validate_precision(self, values: List[float]) -> Dict[str, Any]:
        """
        Validates numerical precision of values.
        
        Args:
            values: List of numerical values to validate
        
        Returns:
            Dictionary with validation results
        """
        self.logger.debug(f"Validating precision for {len(values)} values.")
        
        violations = []
        
        # Check for NaN
        nan_indices = [i for i, v in enumerate(values) if np.isnan(v)]
        if nan_indices:
            violations.append({'type': 'NaN', 'indices': nan_indices})
            self.logger.error(f"NaN values found at indices: {nan_indices[:10]}...")
        
        # Check for Inf
        inf_indices = [i for i, v in enumerate(values) if np.isinf(v)]
        if inf_indices:
            violations.append({'type': 'Inf', 'indices': inf_indices})
            self.logger.error(f"Inf values found at indices: {inf_indices[:10]}...")
        
        # Check precision (for non-NaN, non-Inf values)
        finite_values = [v for v in values if np.isfinite(v)]
        if finite_values:
            # Check if values are within float64 precision
            max_value = max(abs(v) for v in finite_values)
            min_value = min(abs(v) for v in finite_values if v != 0)
            
            # Precision check: relative error should be within tolerance
            if min_value > 0:
                relative_precision = self.precision_tolerance / min_value
                if relative_precision > 1e-15:  # Float64 machine epsilon
                    violations.append({
                        'type': 'Precision',
                        'message': f'Relative precision {relative_precision:.2e} exceeds machine epsilon'
                    })
        
        is_valid = len(violations) == 0
        
        return {
            'is_valid': is_valid,
            'violations': violations,
            'total_values': len(values),
            'finite_values': len(finite_values),
            'precision_tolerance': self.precision_tolerance
        }
    
    def validate_constraint_satisfaction(self, constraint_values: List[float], 
                                       constraint_types: List[str]) -> Dict[str, Any]:
        """
        Validates constraint satisfaction.
        
        Args:
            constraint_values: List of constraint violation values
            constraint_types: List of constraint types ('hard' or 'soft')
        
        Returns:
            Dictionary with validation results
        """
        self.logger.debug(f"Validating {len(constraint_values)} constraints.")
        
        violations = []
        
        for i, (value, ctype) in enumerate(zip(constraint_values, constraint_types)):
            # Hard constraints: must be ≤ 0
            if ctype == 'hard' and value > self.precision_tolerance:
                violations.append({
                    'index': i,
                    'type': 'hard',
                    'value': value,
                    'threshold': 0.0
                })
            # Soft constraints: should be ≤ threshold (e.g., 0.1)
            elif ctype == 'soft' and value > 0.1:
                violations.append({
                    'index': i,
                    'type': 'soft',
                    'value': value,
                    'threshold': 0.1
                })
        
        is_valid = len(violations) == 0
        
        return {
            'is_valid': is_valid,
            'violations': violations,
            'total_constraints': len(constraint_values),
            'hard_constraints': sum(1 for ct in constraint_types if ct == 'hard'),
            'soft_constraints': sum(1 for ct in constraint_types if ct == 'soft')
        }
    
    def validate_objective_bounds(self, objectives: List[float], 
                                 expected_bounds: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Validates that objective values are within expected bounds.
        
        Args:
            objectives: List of objective values
            expected_bounds: List of (lower, upper) bounds for each objective
        
        Returns:
            Dictionary with validation results
        """
        self.logger.debug(f"Validating bounds for {len(objectives)} objectives.")
        
        violations = []
        
        for i, (obj_val, (lower, upper)) in enumerate(zip(objectives, expected_bounds)):
            if obj_val < lower or obj_val > upper:
                violations.append({
                    'objective': i,
                    'value': obj_val,
                    'expected_range': (lower, upper)
                })
                self.logger.warning(f"Objective {i} out of bounds: {obj_val} not in [{lower}, {upper}]")
        
        is_valid = len(violations) == 0
        
        return {
            'is_valid': is_valid,
            'violations': violations,
            'total_objectives': len(objectives)
        }
    
    def validate_numerical_stability(self, values: List[float]) -> Dict[str, Any]:
        """
        Validates numerical stability of calculations.
        
        Args:
            values: List of numerical values
        
        Returns:
            Dictionary with validation results
        """
        self.logger.debug("Validating numerical stability.")
        
        # Check for extreme values
        finite_values = [v for v in values if np.isfinite(v)]
        
        if not finite_values:
            return {
                'is_stable': False,
                'message': 'No finite values found',
                'issues': ['all_values_invalid']
            }
        
        max_val = max(finite_values)
        min_val = min(finite_values)
        
        # Check for overflow/underflow
        issues = []
        
        if max_val > 1e10:
            issues.append('potential_overflow')
            self.logger.warning(f"Large values detected: max = {max_val:.2e}")
        
        if min_val < 1e-10 and min_val > 0:
            issues.append('potential_underflow')
            self.logger.warning(f"Small values detected: min = {min_val:.2e}")
        
        # Check for extreme range
        if max_val > 0 and min_val > 0:
            range_ratio = max_val / min_val
            if range_ratio > 1e12:
                issues.append('extreme_range')
                self.logger.warning(f"Extreme value range: ratio = {range_ratio:.2e}")
        
        is_stable = len(issues) == 0
        
        return {
            'is_stable': is_stable,
            'issues': issues,
            'value_range': (min_val, max_val),
            'total_values': len(values),
            'finite_values': len(finite_values)
        }
    
    def validate_matrix_conditioning(self, matrix: np.ndarray) -> Dict[str, Any]:
        """
        Validates matrix conditioning for numerical stability.
        
        Args:
            matrix: Matrix to validate
        
        Returns:
            Dictionary with validation results
        """
        self.logger.debug("Validating matrix conditioning.")
        
        try:
            # Calculate condition number
            condition_number = np.linalg.cond(matrix)
            
            # Check if matrix is well-conditioned
            is_well_conditioned = condition_number < 1e12
            
            if not is_well_conditioned:
                self.logger.warning(f"Matrix is ill-conditioned: κ = {condition_number:.2e}")
            
            # Check for singularity
            is_singular = np.linalg.matrix_rank(matrix) < min(matrix.shape)
            
            return {
                'is_well_conditioned': is_well_conditioned,
                'condition_number': float(condition_number),
                'is_singular': is_singular,
                'rank': int(np.linalg.matrix_rank(matrix)),
                'shape': matrix.shape
            }
            
        except Exception as e:
            self.logger.error(f"Error validating matrix conditioning: {e}", exc_info=True)
            return {
                'is_well_conditioned': False,
                'error': str(e)
            }
    
    def validate_rounding_errors(self, computed: float, expected: float, 
                                tolerance: float = None) -> Dict[str, Any]:
        """
        Validates rounding errors in computed values.
        
        Args:
            computed: Computed value
            expected: Expected value
            tolerance: Allowed error tolerance (defaults to precision_tolerance)
        
        Returns:
            Dictionary with validation results
        """
        if tolerance is None:
            tolerance = self.precision_tolerance
        
        error = abs(computed - expected)
        relative_error = error / abs(expected) if expected != 0 else error
        
        is_within_tolerance = error <= tolerance
        
        return {
            'is_within_tolerance': is_within_tolerance,
            'absolute_error': float(error),
            'relative_error': float(relative_error),
            'tolerance': tolerance,
            'computed': float(computed),
            'expected': float(expected)
        }


