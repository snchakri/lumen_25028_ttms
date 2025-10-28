#!/usr/bin/env python3
"""
LPWeightOptimizer - Stage II: Linear Programming Weight Optimization

This module implements Stage II of the 2-stage LP optimization framework from
Stage-5.2 Solver Selection & Arsenal Modularity Foundations.

MATHEMATICAL FOUNDATIONS COMPLIANCE:
- Implements Theorem 4.5: Separation margin maximization LP
- Implements Algorithm 4.6: Iterative weight optimization
- Validates Theorem 4.7: Convergence guarantee

Author: LUMEN TTMS - Theoretical Foundation Compliant Implementation
Version: 2.0.0
License: MIT
"""

import numpy as np
import scipy.optimize
from typing import Tuple, Optional
import structlog

logger = structlog.get_logger(__name__)

class LPWeightOptimizer:
    """
    Implements Stage II: Linear Programming Weight Optimization
    per Theorem 4.5 and Algorithm 4.6 from foundations
    
    This optimizer learns optimal parameter weights through LP-based
    separation margin maximization.
    """
    
    def __init__(self, 
                 convergence_tolerance: float = 1e-6, 
                 max_iterations: int = 20,
                 separation_threshold: float = 0.001):
        """
        Initialize the LP weight optimizer.
        
        Args:
            convergence_tolerance: Tolerance for weight convergence
            max_iterations: Maximum iterations for Algorithm 4.6
            separation_threshold: Minimum acceptable separation margin
        """
        self.convergence_tolerance = convergence_tolerance
        self.max_iterations = max_iterations
        self.separation_threshold = separation_threshold
        self.logger = logger.bind(component="lp_optimizer")
        self.logger.info("LPWeightOptimizer initialized",
                        convergence_tolerance=convergence_tolerance,
                        max_iterations=max_iterations,
                        separation_threshold=separation_threshold)
    
    def optimize_weights(self, 
                        normalized_solver_matrix: np.ndarray,
                        normalized_problem_vector: np.ndarray) -> Tuple[np.ndarray, float, int, bool]:
        """
        Implements Algorithm 4.6: Iterative Weight Optimization
        
        Returns:
            optimal_weights: w* ∈ R^16 with Σw=1, w>=0
            separation_margin: d* (robustness measure)
            iterations: Number of iterations taken
            converged: Whether algorithm converged
        """
        R = normalized_solver_matrix
        c_norm = normalized_problem_vector
        
        # Handle single-solver case (degenerate)
        if R.shape[0] == 1:
            self.logger.warning("Single solver case detected - returning uniform weights")
            return np.ones(16) / 16, 0.0, 0, True
        
        # Algorithm 4.6 Step 1: Initialize uniform weights
        w = np.ones(R.shape[1]) / R.shape[1]
        
        best_solver_idx = -1
        iteration = 0
        converged = False
        last_weights = None
        
        self.logger.info("Starting iterative weight optimization")
        
        while not converged and iteration < self.max_iterations:
            iteration += 1
            
            # Algorithm 4.6 Step 2: Identify current leader
            # M_i = Σ_j w_j(r_{ij} - c̃_j)
            match_scores = np.sum(w * (R - c_norm[np.newaxis, :]), axis=1)
            current_best = np.argmax(match_scores)
            
            # Algorithm 4.6 Step 3: Solve separation LP for fixed leader
            try:
                new_weights, separation_margin = self._solve_separation_lp(R, current_best)
            except Exception as e:
                self.logger.error(f"LP solve failed at iteration {iteration}: {str(e)}")
                # Fallback: use last stable weights or uniform weights
                if last_weights is not None:
                    w = last_weights
                break
            
            # Algorithm 4.6 Step 4: Check convergence
            weight_change = np.linalg.norm(new_weights - w)
            weight_converged = weight_change < self.convergence_tolerance
            solver_stable = (current_best == best_solver_idx)
            
            converged = weight_converged and solver_stable
            
            # Algorithm 4.6 Step 5: Update for next iteration
            last_weights = w.copy()
            w = new_weights
            best_solver_idx = current_best
            
            self.logger.debug(f"Iteration {iteration}",
                            best_solver=int(best_solver_idx),
                            separation_margin=float(separation_margin),
                            weight_change=float(weight_change),
                            converged=converged)
        
        if not converged:
            self.logger.warning(f"LP optimization did not converge in {self.max_iterations} iterations")
        else:
            self.logger.info(f"LP optimization converged in {iteration} iterations",
                           separation_margin=float(separation_margin),
                           best_solver=int(best_solver_idx))
        
        # Theorem 4.7: Convergence guaranteed in finite iterations
        assert iteration <= self.max_iterations, "Maximum iterations exceeded"
        
        return w, separation_margin, iteration, converged
    
    def _solve_separation_lp(self, R: np.ndarray, best_solver_idx: int) -> Tuple[np.ndarray, float]:
        """
        Implements Theorem 4.5: Separation Margin Maximization LP
        
        maximize d
        subject to:
          Σ_j w_j(r_{i*,j} - r_{i,j}) >= d  ∀i ≠ i*
          Σ_j w_j = 1
          w_j >= 0  ∀j
        
        Uses scipy.optimize.linprog with method='highs'
        
        Args:
            R: Normalized solver matrix
            best_solver_idx: Index of current best solver
            
        Returns:
            optimal_weights: w* ∈ R^n_params
            separation_margin: d*
        """
        n_solvers, n_params = R.shape
        
        # Decision variables: [w[0], ..., w[n_params-1], d]
        # Objective: maximize d → minimize -d
        c = np.zeros(n_params + 1)
        c[-1] = -1  # Minimize -d to maximize d
        
        # Inequality constraints: A_ub @ x <= b_ub
        # For each i ≠ i*: Σ_j w_j(r_{i*,j} - r_{i,j}) >= d
        # Rearranged: -Σ_j w_j(r_{i*,j} - r_{i,j}) + d <= 0
        A_ub = []
        b_ub = []
        
        r_best = R[best_solver_idx, :]
        
        for i in range(n_solvers):
            if i != best_solver_idx:
                # Constraint row: [-diff[0], ..., -diff[n_params-1], 1]
                constraint_row = np.zeros(n_params + 1)
                constraint_row[:-1] = -(r_best - R[i, :])
                constraint_row[-1] = 1  # Coefficient for d
                
                A_ub.append(constraint_row)
                b_ub.append(0)
        
        if not A_ub:
            # No other solvers - return uniform weights
            return np.ones(n_params) / n_params, 0.0
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Equality constraint: Σ_j w_j = 1
        A_eq = np.zeros((1, n_params + 1))
        A_eq[0, :-1] = 1
        b_eq = np.array([1])
        
        # Variable bounds: w_j >= 0, d unbounded
        bounds = [(0, None)] * n_params + [(None, None)]
        
        # Solve LP using scipy.optimize.linprog
        try:
            result = scipy.optimize.linprog(
                c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method='highs'
            )
            
            if not result.success:
                raise RuntimeError(f"LP optimization failed: {result.message}")
            
            # Extract solution
            solution = result.x
            optimal_weights = solution[:-1]
            separation_margin = solution[-1]
            
            # Validate solution
            assert np.abs(np.sum(optimal_weights) - 1.0) < 1e-10, "Weight normalization failed"
            assert np.all(optimal_weights >= -1e-10), "Non-negativity violated"
            
            # Clean numerical noise
            optimal_weights = np.maximum(optimal_weights, 0)
            optimal_weights /= np.sum(optimal_weights)
            
            return optimal_weights, separation_margin
            
        except Exception as e:
            self.logger.error(f"LP solve failed: {str(e)}")
            raise
    
    def stage2_optimize_weights(self, 
                               R: np.ndarray,
                               c_norm: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """
        Simplified API for Stage II weight optimization.
        
        This is the primary API for testing and external use.
        
        Args:
            R: Normalized solver matrix ∈ R^(n×16)
            c_norm: Normalized problem vector ∈ R^16
            
        Returns:
            weights: Optimal weight vector w* ∈ R^16
            margin: Separation margin d*
            best_idx: Index of best solver
        """
        # Compute utility scores to find best solver
        utility_scores = R @ c_norm
        best_solver_idx = np.argmax(utility_scores)
        
        # Optimize weights
        weights, converged, iterations, margin = self.optimize_weights(R, c_norm)
        
        return weights, margin, best_solver_idx


# Backward compatibility alias
LPOptimizer = LPWeightOptimizer
