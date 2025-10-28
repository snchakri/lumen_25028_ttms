#!/usr/bin/env python3
"""
SolverSelectionEngine - Complete 2-Stage LP Framework Implementation

This module implements the complete 2-stage LP optimization framework from
Stage-5.2 Solver Selection & Arsenal Modularity Foundations.

MATHEMATICAL FOUNDATIONS COMPLIANCE:
- Stage I: L2 normalization per Definitions 3.1-3.2
- Stage II: LP weight optimization per Theorem 4.5 and Algorithm 4.6
- NO ensemble voting, NO heuristic selection
- Pure mathematical optimization

Author: LUMEN TTMS - Theoretical Foundation Compliant Implementation
Version: 2.0.0
License: MIT
"""

import structlog
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import numpy as np

from .. import Stage5Configuration
from .normalization_engine import L2NormalizationEngine
from .lp_optimizer import LPWeightOptimizer

logger = structlog.get_logger(__name__)

@dataclass
class SolverCapability:
    """Represents solver capability information from JSON."""
    solver_id: str
    name: str
    capability_vector: List[float]  # 16 parameters
    limits: Dict[str, Any]
    performance_profile: Dict[str, Any]
    deployment_info: Dict[str, Any]

@dataclass
class SolverSelectionResult:
    """Result of solver selection process."""
    selected_solver_id: str
    selected_solver: SolverCapability
    optimal_weights: np.ndarray
    separation_margin: float
    confidence: float
    all_match_scores: np.ndarray
    iteration_count: int
    converged: bool
    selection_metadata: Dict[str, Any]

class SolverSelectionEngine:
    """
    Complete 2-stage LP framework for solver selection.
    
    Implements the mathematical framework from Stage-5.2 foundations:
    - Stage I: L2 normalization
    - Stage II: LP weight optimization
    - NO ensemble voting, NO heuristics
    """
    
    def __init__(self, config: Optional[Stage5Configuration] = None):
        """
        Initialize the solver selection engine.
        
        Args:
            config: Optional configuration for the engine
        """
        self.config = config or Stage5Configuration()
        self.logger = logger.bind(component="solver_selection_engine")
        
        # Initialize 2-stage LP framework components
        self.normalization_engine = L2NormalizationEngine(
            epsilon=self.config.l2_norm_epsilon
        )
        self.lp_optimizer = LPWeightOptimizer(
            convergence_tolerance=self.config.lp_convergence_tolerance,
            max_iterations=self.config.lp_max_iterations,
            separation_threshold=self.config.separation_margin_threshold
        )
        
        # Solver capabilities storage
        self.solver_capabilities = []
        
        self.logger.info("SolverSelectionEngine initialized with 2-stage LP framework",
                        lp_convergence_tolerance=self.config.lp_convergence_tolerance,
                        lp_max_iterations=self.config.lp_max_iterations)
    
    def load_solver_capabilities(self, capabilities_path: Union[str, Path]) -> None:
        """
        Load solver capabilities from JSON file.
        
        Args:
            capabilities_path: Path to solver_capabilities.json
        """
        capabilities_path = Path(capabilities_path)
        
        if not capabilities_path.exists():
            raise FileNotFoundError(f"Solver capabilities file not found: {capabilities_path}")
        
            with open(capabilities_path, 'r') as f:
                capabilities_data = json.load(f)
            
        self.solver_capabilities = []
        
        for solver_data in capabilities_data.get('solvers', []):
            capability = SolverCapability(
                solver_id=solver_data['solver_id'],
                name=solver_data['name'],
                capability_vector=solver_data['capability_vector'],
                limits=solver_data.get('limits', {}),
                performance_profile=solver_data.get('performance_profile', {}),
                deployment_info=solver_data.get('deployment_info', {})
            )
            self.solver_capabilities.append(capability)
        
        self.logger.info(f"Loaded {len(self.solver_capabilities)} solver capabilities",
                        solvers=[s.solver_id for s in self.solver_capabilities])
    
    def select_optimal_solver(self, 
                             complexity_vector: np.ndarray,
                             solver_capabilities_path: Optional[Union[str, Path]] = None) -> SolverSelectionResult:
        """
        Execute complete 2-stage LP framework for solver selection.
        
        Args:
            complexity_vector: 16-parameter complexity vector from Stage 5.1
            solver_capabilities_path: Path to solver_capabilities.json (if not already loaded)
            
        Returns:
            SolverSelectionResult with selected solver and metadata
        """
        start_time = time.time()
        
        self.logger.info("Starting 2-stage LP solver selection",
                        complexity_vector_shape=complexity_vector.shape)
        
        # Load solver capabilities if not already loaded
        if not self.solver_capabilities and solver_capabilities_path:
            self.load_solver_capabilities(solver_capabilities_path)
        
        if not self.solver_capabilities:
            raise ValueError("No solver capabilities available")
        
        # Build capability matrix X ∈ R^(n×16)
        capability_matrix = self._build_capability_matrix()
        
        # Stage I: L2 Normalization
        self.logger.info("Stage I: L2 Normalization")
        R, c_norm, norm_factors = self.normalization_engine.normalize(
            capability_matrix, complexity_vector
        )
        
        # Handle zero-variance parameters (fallback)
        R_filtered, c_norm_filtered, factors_filtered = self.normalization_engine.handle_zero_variance_parameters(
            R, c_norm, norm_factors
        )
        
        # Stage II: LP Weight Optimization
        self.logger.info("Stage II: LP Weight Optimization")
        w_optimal, separation_margin, iterations, converged = self.lp_optimizer.optimize_weights(
            R_filtered, c_norm_filtered
        )
        
        # Pad weights back to 16 dimensions if any parameters were removed
        if w_optimal.shape[0] < 16:
            w_padded = np.zeros(16)
            w_padded[factors_filtered > self.config.l2_norm_epsilon] = w_optimal
            w_optimal = w_padded
        
        # Stage III: Final Selection
        self.logger.info("Stage III: Final Selection")
        final_match_scores = np.sum(w_optimal * (R - c_norm[np.newaxis, :]), axis=1)
        best_idx = np.argmax(final_match_scores)
        
        selected_solver = self.solver_capabilities[best_idx]
        
        # Calculate confidence
        confidence = separation_margin / (np.max(final_match_scores) + 1e-10)
        confidence = min(1.0, max(0.0, confidence))
        
        # Confidence gating (fallback mechanism)
        if confidence < self.config.solver_confidence_threshold:
            self.logger.warning(f"Low confidence {confidence:.4f} < {self.config.solver_confidence_threshold}",
                              selected_solver=selected_solver.solver_id)
        
        processing_time = time.time() - start_time
        
        # Create selection metadata
        selection_metadata = {
            "stage1_normalization": {
                "n_solvers": R.shape[0],
                "n_parameters": R.shape[1],
                "normalization_factors_range": [float(norm_factors.min()), float(norm_factors.max())]
            },
            "stage2_lp_optimization": {
                "iterations": iterations,
                "converged": converged,
                "separation_margin": float(separation_margin),
                "optimal_weights": w_optimal.tolist()
            },
            "stage3_final_selection": {
                "selected_solver_idx": int(best_idx),
                "match_score": float(final_match_scores[best_idx]),
                "confidence": float(confidence),
                "all_match_scores": final_match_scores.tolist()
            },
            "processing_time_seconds": processing_time
        }
        
        result = SolverSelectionResult(
            selected_solver_id=selected_solver.solver_id,
            selected_solver=selected_solver,
            optimal_weights=w_optimal,
            separation_margin=separation_margin,
            confidence=confidence,
            all_match_scores=final_match_scores,
            iteration_count=iterations,
            converged=converged,
            selection_metadata=selection_metadata
        )
        
        self.logger.info("2-stage LP solver selection completed",
                        selected_solver=selected_solver.solver_id,
                        confidence=confidence,
                        separation_margin=separation_margin,
                        iterations=iterations,
                        converged=converged,
                        processing_time=processing_time)
        
        return result
    
    def _build_capability_matrix(self) -> np.ndarray:
        """
        Build capability matrix X ∈ R^(n×16) from solver capabilities.
            
        Returns:
            Capability matrix with n solvers and 16 parameters
        """
        if not self.solver_capabilities:
            raise ValueError("No solver capabilities available")
        
        capability_vectors = []
        for solver in self.solver_capabilities:
            capability_vectors.append(solver.capability_vector)
        
        matrix = np.array(capability_vectors)
        
        # Validate dimensions
        if matrix.shape[1] != 16:
            raise ValueError(f"Expected 16 parameters, got {matrix.shape[1]}")
        
        self.logger.debug(f"Built capability matrix: {matrix.shape[0]} solvers × {matrix.shape[1]} parameters")
        
        return matrix
    
    def create_solver_capabilities_template(self, output_path: Union[str, Path]) -> None:
        """
        Create a template solver_capabilities.json file.
        
        Args:
            output_path: Path where template will be written
        """
        template = {
            "version": "1.0",
            "description": "Solver capabilities for Stage 5.2 LP framework",
            "solvers": [
                {
                    "solver_id": "pulp_cbc",
                    "name": "PuLP with CBC",
                    "capability_vector": [0.7, 0.8, 0.6, 0.5, 0.9, 0.7, 0.5, 0.8, 0.4, 0.3, 0.2, 0.6, 0.8, 0.7, 0.5, 0.4],
                    "limits": {
                        "max_variables": 1000000,
                        "max_constraints": 1000000,
                        "memory_mb": 4096
                    },
                    "performance_profile": {
                        "typical_solve_time_seconds": 120,
                        "convergence_rate": 0.95
                    },
                    "deployment_info": {
                        "module_path": "pulp",
                        "entry_function": "solve",
                        "backend_enum": "CBC"
                    }
                }
            ]
        }
        
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        self.logger.info(f"Created solver capabilities template at {output_path}")
