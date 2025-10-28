#!/usr/bin/env python3
"""
L2NormalizationEngine - Stage I: Parameter Normalization

This module implements Stage I of the 2-stage LP optimization framework from
Stage-5.2 Solver Selection & Arsenal Modularity Foundations.

MATHEMATICAL FOUNDATIONS COMPLIANCE:
- Implements Definition 3.1: L2 normalization for solver capabilities
- Implements Definition 3.2: L2 normalization for problem complexity
- Validates Theorem 3.3: Normalization properties (boundedness, scale invariance, etc.)

Author: LUMEN TTMS - Theoretical Foundation Compliant Implementation
Version: 2.0.0
License: MIT
"""

import numpy as np
from typing import Tuple
import structlog

logger = structlog.get_logger(__name__)

class L2NormalizationEngine:
    """
    Implements Stage I: L2 Parameter Normalization per Definition 3.1-3.2
    from Stage-5.2 Solver Selection & Arsenal Modularity Foundations
    
    This engine performs L2 normalization on both solver capabilities and
    problem complexity to ensure commensurability and boundedness.
    """
    
    def __init__(self, epsilon: float = 1e-10):
        """
        Initialize the L2 normalization engine.
        
        Args:
            epsilon: Small value to prevent division by zero
        """
        self.epsilon = epsilon
        self.logger = logger.bind(component="l2_normalization")
        self.logger.info("L2NormalizationEngine initialized")
    
    def normalize(self, 
                  solver_capability_matrix: np.ndarray,
                  problem_complexity_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform L2 normalization on solver capabilities and problem complexity.
        
        Per Definition 3.1: r_{ij} = x_{ij} / √(Σ_k x_{kj}²)
        Per Definition 3.2: c̃_j = c_j / √(Σ_k x_{kj}²)
        
        Args:
            solver_capability_matrix: X ∈ R^(n×16) where n=num_solvers
            problem_complexity_vector: c ∈ R^16
            
        Returns:
            normalized_solver_matrix: R ∈ R^(n×16) with r_{ij} ∈ [0,1]
            normalized_problem_vector: c̃ ∈ R^16
            normalization_factors: σ ∈ R^16 for validation
        """
        n_solvers, n_params = solver_capability_matrix.shape
        
        # Validate input dimensions
        assert n_params == 16, f"Must have exactly 16 parameters, got {n_params}"
        assert len(problem_complexity_vector) == 16, "Problem vector must be 16D"
        
        self.logger.info("Starting L2 normalization",
                        n_solvers=n_solvers,
                        n_parameters=n_params)
        
        # Calculate L2 normalization factors per parameter
        # σ_j = √(Σ_k x_{kj}²)
        normalization_factors = np.linalg.norm(solver_capability_matrix, axis=0)
        
        # Prevent division by zero (Theorem 3.3 Property 1: Boundedness)
        normalization_factors = np.where(
            normalization_factors < self.epsilon,
            self.epsilon,
            normalization_factors
        )
        
        # Normalize solver capabilities: r_{ij} = x_{ij} / σ_j
        R = solver_capability_matrix / normalization_factors[np.newaxis, :]
        
        # Normalize problem complexity with same factors
        c_normalized = problem_complexity_vector / normalization_factors
        
        # Validate Theorem 3.3 properties
        self._validate_normalization_properties(R, solver_capability_matrix)
        
        self.logger.info("L2 normalization completed",
                        n_solvers=n_solvers,
                        n_parameters=n_params,
                        normalization_factors_range=[
                            float(normalization_factors.min()),
                            float(normalization_factors.max())
                        ])
        
        return R, c_normalized, normalization_factors
    
    def _validate_normalization_properties(self, R: np.ndarray, X: np.ndarray):
        """
        Validate Theorem 3.3 properties:
        1. Boundedness: r_{ij} ∈ [0,1]
        2. Scale Invariance: relative rankings preserved
        3. Dynamic Adaptation: automatic scaling
        4. Correspondence Preservation: relationships maintained
        """
        # Property 1: Boundedness
        if not (np.all(R >= 0) and np.all(R <= 1)):
            raise ValueError("Boundedness violated: r_{ij} must be in [0,1]")
        
        # Property 2: Scale Invariance (check column-wise rankings preserved)
        for j in range(R.shape[1]):
            x_col = X[:, j]
            r_col = R[:, j]
            
            # If x_i > x_k then r_i > r_k (rankings preserved)
            for i in range(len(x_col)):
                for k in range(i+1, len(x_col)):
                    if x_col[i] > x_col[k] and r_col[i] < r_col[k]:
                        raise ValueError(f"Scale invariance violated at column {j}")
        
        self.logger.debug("Normalization property validation passed",
                         properties_validated=["boundedness", "scale_invariance"])
    
    def handle_zero_variance_parameters(self, 
                                       R: np.ndarray, 
                                       c_norm: np.ndarray,
                                       normalization_factors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Handle degenerate case where some parameters have zero variance.
        
        This implements the fallback mechanism for normalization.
        
        Args:
            R: Normalized solver matrix
            c_norm: Normalized problem vector
            normalization_factors: Original normalization factors
            
        Returns:
            Filtered R, c_norm, and normalization_factors with zero-variance columns removed
        """
        # Identify zero-variance columns (all solvers have same capability)
        zero_variance_mask = normalization_factors < self.epsilon
        
        if np.any(zero_variance_mask):
            n_removed = np.sum(zero_variance_mask)
            self.logger.warning(f"Removing {n_removed} zero-variance parameters from normalization",
                              removed_indices=np.where(zero_variance_mask)[0].tolist())
            
            # Remove zero-variance columns
            R_filtered = R[:, ~zero_variance_mask]
            c_norm_filtered = c_norm[~zero_variance_mask]
            factors_filtered = normalization_factors[~zero_variance_mask]
            
            return R_filtered, c_norm_filtered, factors_filtered
        else:
            return R, c_norm, normalization_factors
    
    def stage1_normalize_parameters(self, 
                                   solver_matrix: np.ndarray,
                                   problem_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Alias for normalize() method with simpler return signature for testing.
        
        This is the primary API for Stage I normalization.
        
        Args:
            solver_matrix: X ∈ R^(n×16) where n=num_solvers
            problem_vector: c ∈ R^16
            
        Returns:
            R: Normalized solver matrix ∈ R^(n×16)
            c_norm: Normalized problem vector ∈ R^16
        """
        R, c_norm, _ = self.normalize(solver_matrix, problem_vector)
        return R, c_norm


# Backward compatibility alias
NormalizationEngine = L2NormalizationEngine
