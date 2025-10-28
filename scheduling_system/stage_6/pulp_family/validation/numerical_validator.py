"""
Numerical Validator - Definition 11.3 Compliance

Validates solution accuracy bounds using scipy for numerical analysis.

Compliance:
- Definition 11.3: Solution Accuracy Bounds
- ||Ax* - b|| ≤ ε_feasibility
- ||c^T x* - z*|| ≤ ε_optimality

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from scipy.linalg import norm
from scipy.sparse import issparse


@dataclass
class NumericalValidationResult:
    """Result of numerical validation."""
    
    check_name: str
    passed: bool
    computed_value: float
    tolerance: float
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'check_name': self.check_name,
            'passed': self.passed,
            'computed_value': self.computed_value,
            'tolerance': self.tolerance,
            'details': self.details
        }


class NumericalValidator:
    """
    Validates numerical accuracy per Definition 11.3.
    
    Uses scipy for numerical computations.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize numerical validator."""
        self.logger = logger or logging.getLogger(__name__)
    
    def validate_feasibility_accuracy(
        self,
        constraint_matrix: np.ndarray,
        solution_vector: np.ndarray,
        constraint_bounds: np.ndarray,
        tolerance: float = 1e-6
    ) -> NumericalValidationResult:
        """
        Validate feasibility accuracy: ||Ax* - b|| ≤ ε_feasibility.
        
        Compliance: Definition 11.3
        
        Args:
            constraint_matrix: Matrix A
            solution_vector: Solution x*
            constraint_bounds: Bounds b
            tolerance: Feasibility tolerance ε_feasibility
        
        Returns:
            NumericalValidationResult
        """
        self.logger.info("Validating feasibility accuracy per Definition 11.3...")
        
        try:
            # Handle sparse matrices
            if issparse(constraint_matrix):
                residual = constraint_matrix.dot(solution_vector) - constraint_bounds
            else:
                residual = np.dot(constraint_matrix, solution_vector) - constraint_bounds
            
            # Compute norm of residual
            residual_norm = norm(residual)
            
            # Check if within tolerance
            passed = residual_norm <= tolerance
            
            details = {
                'residual_norm': residual_norm,
                'max_residual': np.max(np.abs(residual)),
                'constraint_matrix_shape': constraint_matrix.shape,
                'solution_vector_size': len(solution_vector),
                'constraint_bounds_size': len(constraint_bounds)
            }
            
            self.logger.info(f"Feasibility residual norm: {residual_norm:.2e} (tolerance: {tolerance:.2e})")
            
            return NumericalValidationResult(
                check_name="Feasibility Accuracy",
                passed=passed,
                computed_value=residual_norm,
                tolerance=tolerance,
                details=details
            )
            
        except Exception as e:
            self.logger.error(f"Error validating feasibility accuracy: {str(e)}")
            
            return NumericalValidationResult(
                check_name="Feasibility Accuracy",
                passed=False,
                computed_value=float('inf'),
                tolerance=tolerance,
                details={'error': str(e)}
            )
    
    def validate_optimality_accuracy(
        self,
        objective_coefficients: np.ndarray,
        solution_vector: np.ndarray,
        reported_objective: float,
        tolerance: float = 1e-6
    ) -> NumericalValidationResult:
        """
        Validate optimality accuracy: ||c^T x* - z*|| ≤ ε_optimality.
        
        Compliance: Definition 11.3
        
        Args:
            objective_coefficients: Coefficients c
            solution_vector: Solution x*
            reported_objective: Reported objective z*
            tolerance: Optimality tolerance ε_optimality
        
        Returns:
            NumericalValidationResult
        """
        self.logger.info("Validating optimality accuracy per Definition 11.3...")
        
        try:
            # Compute objective value from solution
            computed_objective = np.dot(objective_coefficients, solution_vector)
            
            # Compute difference from reported objective
            objective_error = abs(computed_objective - reported_objective)
            
            # Check if within tolerance
            passed = objective_error <= tolerance
            
            details = {
                'computed_objective': computed_objective,
                'reported_objective': reported_objective,
                'objective_error': objective_error,
                'relative_error': objective_error / abs(reported_objective) if reported_objective != 0 else float('inf'),
                'objective_coefficients_size': len(objective_coefficients),
                'solution_vector_size': len(solution_vector)
            }
            
            self.logger.info(f"Objective error: {objective_error:.2e} (tolerance: {tolerance:.2e})")
            
            return NumericalValidationResult(
                check_name="Optimality Accuracy",
                passed=passed,
                computed_value=objective_error,
                tolerance=tolerance,
                details=details
            )
            
        except Exception as e:
            self.logger.error(f"Error validating optimality accuracy: {str(e)}")
            
            return NumericalValidationResult(
                check_name="Optimality Accuracy",
                passed=False,
                computed_value=float('inf'),
                tolerance=tolerance,
                details={'error': str(e)}
            )
    
    def validate_solution_bounds(
        self,
        solution_vector: np.ndarray,
        variable_bounds: Dict[str, Tuple[float, float]]
    ) -> NumericalValidationResult:
        """
        Validate solution respects variable bounds.
        
        Args:
            solution_vector: Solution x*
            variable_bounds: Variable bounds {var_name: (lower, upper)}
        
        Returns:
            NumericalValidationResult
        """
        self.logger.info("Validating solution bounds...")
        
        try:
            violations = []
            max_violation = 0.0
            
            for i, (var_name, (lower, upper)) in enumerate(variable_bounds.items()):
                if i < len(solution_vector):
                    value = solution_vector[i]
                    
                    if value < lower:
                        violation = lower - value
                        violations.append(f"{var_name}: {value} < {lower} (violation: {violation})")
                        max_violation = max(max_violation, violation)
                    elif value > upper:
                        violation = value - upper
                        violations.append(f"{var_name}: {value} > {upper} (violation: {violation})")
                        max_violation = max(max_violation, violation)
            
            passed = len(violations) == 0
            
            details = {
                'n_violations': len(violations),
                'max_violation': max_violation,
                'violations': violations[:10],  # First 10 violations
                'total_variables': len(variable_bounds)
            }
            
            self.logger.info(f"Bound violations: {len(violations)} (max violation: {max_violation:.2e})")
            
            return NumericalValidationResult(
                check_name="Solution Bounds",
                passed=passed,
                computed_value=max_violation,
                tolerance=0.0,
                details=details
            )
            
        except Exception as e:
            self.logger.error(f"Error validating solution bounds: {str(e)}")
            
            return NumericalValidationResult(
                check_name="Solution Bounds",
                passed=False,
                computed_value=float('inf'),
                tolerance=0.0,
                details={'error': str(e)}
            )
    
    def validate_integer_constraints(
        self,
        solution_vector: np.ndarray,
        integer_variables: set,
        tolerance: float = 1e-6
    ) -> NumericalValidationResult:
        """
        Validate integer variables are actually integer.
        
        Args:
            solution_vector: Solution x*
            integer_variables: Set of integer variable indices
            tolerance: Integer tolerance
        
        Returns:
            NumericalValidationResult
        """
        self.logger.info("Validating integer constraints...")
        
        try:
            violations = []
            max_violation = 0.0
            
            for i in integer_variables:
                if i < len(solution_vector):
                    value = solution_vector[i]
                    # Round to nearest integer for comparison (exact operation)
                    rounded_value = round(value)
                    # Check if value is sufficiently close to integer (within tolerance)
                    violation = abs(value - rounded_value)
                    
                    if violation > tolerance:
                        violations.append(f"Variable {i}: {value} (should be integer)")
                        max_violation = max(max_violation, violation)
            
            passed = len(violations) == 0
            
            details = {
                'n_violations': len(violations),
                'max_violation': max_violation,
                'violations': violations[:10],  # First 10 violations
                'total_integer_variables': len(integer_variables)
            }
            
            self.logger.info(f"Integer violations: {len(violations)} (max violation: {max_violation:.2e})")
            
            return NumericalValidationResult(
                check_name="Integer Constraints",
                passed=passed,
                computed_value=max_violation,
                tolerance=tolerance,
                details=details
            )
            
        except Exception as e:
            self.logger.error(f"Error validating integer constraints: {str(e)}")
            
            return NumericalValidationResult(
                check_name="Integer Constraints",
                passed=False,
                computed_value=float('inf'),
                tolerance=tolerance,
                details={'error': str(e)}
            )
    
    def validate_all_numerical_properties(
        self,
        constraint_matrix: np.ndarray,
        solution_vector: np.ndarray,
        constraint_bounds: np.ndarray,
        objective_coefficients: np.ndarray,
        reported_objective: float,
        variable_bounds: Dict[str, Tuple[float, float]],
        integer_variables: set,
        feasibility_tolerance: float = 1e-6,
        optimality_tolerance: float = 1e-6
    ) -> Dict[str, NumericalValidationResult]:
        """
        Validate all numerical properties per Definition 11.3.
        
        Args:
            constraint_matrix: Matrix A
            solution_vector: Solution x*
            constraint_bounds: Bounds b
            objective_coefficients: Coefficients c
            reported_objective: Reported objective z*
            variable_bounds: Variable bounds
            integer_variables: Integer variable indices
            feasibility_tolerance: Feasibility tolerance
            optimality_tolerance: Optimality tolerance
        
        Returns:
            Dictionary of validation results
        """
        self.logger.info("Validating all numerical properties per Definition 11.3...")
        
        results = {}
        
        # Feasibility accuracy
        results['feasibility'] = self.validate_feasibility_accuracy(
            constraint_matrix,
            solution_vector,
            constraint_bounds,
            feasibility_tolerance
        )
        
        # Optimality accuracy
        results['optimality'] = self.validate_optimality_accuracy(
            objective_coefficients,
            solution_vector,
            reported_objective,
            optimality_tolerance
        )
        
        # Solution bounds
        results['bounds'] = self.validate_solution_bounds(
            solution_vector,
            variable_bounds
        )
        
        # Integer constraints
        if integer_variables:
            results['integer'] = self.validate_integer_constraints(
                solution_vector,
                integer_variables,
                feasibility_tolerance
            )
        
        # Summary
        passed_count = sum(1 for r in results.values() if r.passed)
        total_count = len(results)
        
        self.logger.info(f"Numerical validation complete: {passed_count}/{total_count} checks passed")
        
        return results

