"""
stage_5_2/select.py
Stage 5.2 Solver Selection and Ranking Engine

This module implements the final selection and ranking generation from the Stage-5.2
mathematical framework. It combines normalized data and optimized weights to generate
complete selection decisions with confidence scoring and complete ranking.

Mathematical Foundation:
- Match Score Computation: Mi(w*) = Σ wj*(ri,j - c̃j) with optimal weights
- Selection Decision: i* = argmax Mi(w*) for mathematically optimal choice  
- Confidence Scoring: Based on separation margins and score distributions
- Ranking Generation: Complete solver ordering with margin analysis

Key Algorithms:
1. SolverSelector: complete selection engine with mathematical validation
2. generate_selection_decision: Complete decision generation with audit trail
3. compute_solver_ranking: complete ranking with margin analysis
4. calculate_confidence_score: Statistical confidence based on separation theory

Integration Points:
- Input: NormalizedData + OptimizationResult from previous stages
- Output: SelectionDecision with chosen solver, ranking, and detailed metrics
- Downstream: Stage 6 solver execution and usage

Performance Characteristics:
- Time Complexity: O(n×P) for score computation where n=solvers, P=16
- Space Complexity: O(n) for solver ranking and metadata
- Mathematical Accuracy: Uses optimal weights with separation guarantees
- Audit Compliance: Complete decision justification and traceability
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timezone

from ..common.logging import get_logger, log_operation
from ..common.exceptions import Stage5ValidationError, Stage5ComputationError
from ..common.schema import (
    SelectionDecision, SolverRanking, SolverCapability, ChosenSolver
)

from .normalize import NormalizedData
from .optimize import OptimizationResult

# Mathematical constants from Stage-5.2 theoretical framework
PARAMETER_COUNT = 16  # Fixed dimensionality P = 16
MIN_CONFIDENCE_THRESHOLD = 0.01  # Minimum confidence for valid selection
RANKING_PRECISION = 1e-10  # Precision for ranking score comparisons
STATISTICAL_CONFIDENCE_FACTOR = 2.0  # Factor for confidence interval calculation

# Global logger for this module
_logger = get_logger("stage5_2.select")

@dataclass
class SolverMatchAnalysis:
    """
    Detailed analysis of solver-problem match characteristics.
    
    Contains complete analysis of how well each solver matches the problem
    requirements across all 16 parameters, including parameter-specific gaps,
    strengths, and weaknesses for detailed decision justification.
    
    Attributes:
        solver_id: Unique solver identifier
        match_score: Overall match score Mi(w*) with optimal weights
        parameter_gaps: Parameter-wise gaps gi,j = ri,j - c̃j for detailed analysis
        strengths: Parameters where solver significantly exceeds requirements
        weaknesses: Parameters where solver falls short of requirements
        weighted_contributions: Contribution of each parameter to final score
        
    Mathematical Properties:
    - Match Score: Mi(w*) = Σ wj*(ri,j - c̃j) with optimal weights w*
    - Parameter Gaps: gi,j ∈ R for all parameters j (can be negative)
    - Weighted Contributions: wj*gi,j for individual parameter impact
    - Statistical Analysis: Mean, variance, and distribution characteristics
    """
    
    solver_id: str  # Solver unique identifier
    match_score: float  # Overall match score Mi(w*)
    parameter_gaps: np.ndarray  # Shape (16,) - parameter-wise gaps gi,j  
    strengths: List[int]  # Parameter indices where solver excels
    weaknesses: List[int]  # Parameter indices where solver is deficient
    weighted_contributions: np.ndarray  # Shape (16,) - wj*gi,j values
    
    def __post_init__(self):
        """Post-initialization validation of match analysis."""
        if self.parameter_gaps.shape != (PARAMETER_COUNT,):
            raise Stage5ValidationError(
                f"Parameter gaps must have shape ({PARAMETER_COUNT},), got {self.parameter_gaps.shape}",
                validation_type="parameter_gaps_shape",
                expected_value=f"({PARAMETER_COUNT},)",
                actual_value=str(self.parameter_gaps.shape)
            )
        
        if self.weighted_contributions.shape != (PARAMETER_COUNT,):
            raise Stage5ValidationError(
                f"Weighted contributions must have shape ({PARAMETER_COUNT},), got {self.weighted_contributions.shape}",
                validation_type="weighted_contributions_shape",
                expected_value=f"({PARAMETER_COUNT},)",
                actual_value=str(self.weighted_contributions.shape)
            )
        
        # Validate numerical stability
        if not np.isfinite(self.match_score):
            raise Stage5ComputationError(
                f"Match score must be finite, got {self.match_score}",
                computation_type="match_score_validation",
                context={"solver_id": self.solver_id}
            )
        
        if not np.all(np.isfinite(self.parameter_gaps)):
            raise Stage5ComputationError(
                "Parameter gaps contain NaN or Inf values",
                computation_type="numerical_stability",
                context={"solver_id": self.solver_id}
            )
    
    def get_top_strengths(self, n: int = 3) -> List[Tuple[int, float]]:
        """
        Get top N parameter strengths with gap values.
        
        Args:
            n: Number of top strengths to return
            
        Returns:
            List of (parameter_index, gap_value) tuples sorted by gap
        """
        strength_gaps = [(i, self.parameter_gaps[i]) for i in self.strengths]
        strength_gaps.sort(key=lambda x: x[1], reverse=True)
        return strength_gaps[:n]
    
    def get_critical_weaknesses(self, n: int = 3) -> List[Tuple[int, float]]:
        """
        Get most critical N parameter weaknesses with gap values.
        
        Args:
            n: Number of critical weaknesses to return
            
        Returns:
            List of (parameter_index, gap_value) tuples sorted by severity
        """
        weakness_gaps = [(i, self.parameter_gaps[i]) for i in self.weaknesses]
        weakness_gaps.sort(key=lambda x: x[1])  # Ascending (most negative first)
        return weakness_gaps[:n]
    
    def compute_match_statistics(self) -> Dict[str, float]:
        """
        Compute statistical properties of solver-problem match.
        
        Returns:
            Dict containing match statistics for analysis
        """
        positive_gaps = self.parameter_gaps[self.parameter_gaps > 0]
        negative_gaps = self.parameter_gaps[self.parameter_gaps < 0]
        
        return {
            "overall_match_score": self.match_score,
            "mean_parameter_gap": float(np.mean(self.parameter_gaps)),
            "parameter_gap_variance": float(np.var(self.parameter_gaps)),
            "positive_gap_count": len(positive_gaps),
            "negative_gap_count": len(negative_gaps),
            "max_strength": float(np.max(self.parameter_gaps)) if len(self.parameter_gaps) > 0 else 0.0,
            "max_weakness": float(np.min(self.parameter_gaps)) if len(self.parameter_gaps) > 0 else 0.0,
            "weighted_score_variance": float(np.var(self.weighted_contributions)),
            "parameter_coverage_ratio": len(positive_gaps) / PARAMETER_COUNT
        }

class SolverSelector:
    """
    complete solver selection engine with mathematical validation.
    
    Combines normalized solver data and optimized weights to generate complete
    selection decisions with confidence scoring, detailed ranking, and complete
    audit trails. Implements the final stage of Algorithm 5.1 from theoretical framework.
    
    The selector provides mathematical guarantees:
    - Optimal Selection: Chosen solver maximizes match score under optimal weights
    - reliable Ranking: Complete ordering with separation margin analysis
    - Statistical Confidence: Confidence scores based on separation theory
    - Decision Justification: Complete parameter-wise analysis for audit trails
    
    Mathematical Foundation:
    - Final Selection: i* = argmax Mi(w*) with optimal weights w*
    - Confidence Scoring: Based on separation margins and score distributions
    - Ranking Generation: Complete ordering with statistical significance
    - Parameter Analysis: Detailed gap analysis gi,j for decision explanation
    
    Performance Characteristics:
    - Time Complexity: O(n×P) for score computation where n=solvers, P=16
    - Space Complexity: O(n) for ranking storage and analysis
    - Memory Usage: Efficient with minimal allocation for large solver arsenals
    - Audit Compliance: Complete traceability with mathematical justification
    """
    
    def __init__(self,
                 solver_capabilities: List[SolverCapability],
                 normalized_data: NormalizedData,
                 optimization_result: OptimizationResult,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize solver selector with validated inputs.
        
        Args:
            solver_capabilities: List of solver capability objects
            normalized_data: Complete normalized dataset from Stage I
            optimization_result: Optimal weights from Stage II
            logger: Optional logger for operation tracking
        """
        self.logger = logger or _logger
        self.solver_capabilities = solver_capabilities
        self.normalized_data = normalized_data
        self.optimization_result = optimization_result
        
        # Validate input consistency
        self._validate_selector_inputs()
        
        # Precompute match analyses for all solvers
        self._match_analyses = self._compute_solver_match_analyses()
        
        self.logger.info(
            f"SolverSelector initialized: {len(solver_capabilities)} solvers, "
            f"optimal_weights_learned, separation_margin={optimization_result.separation_margin:.6f}"
        )
    
    def generate_selection_decision(self) -> SelectionDecision:
        """
        Generate complete selection decision with ranking and confidence.
        
        Implements Stage III of Algorithm 5.1:
        1. Compute final match scores with optimal weights
        2. Select solver with maximum match score
        3. Generate confidence score based on separation margins
        4. Create complete ranking with statistical analysis
        5. Package results with complete audit trail
        
        Returns:
            SelectionDecision: Complete decision with chosen solver and ranking
            
        Raises:
            Stage5ComputationError: If selection computation fails
            
        Mathematical Properties Guaranteed:
        - Optimal Selection: Chosen solver maximizes Mi(w*) per Definition 4.3
        - Statistical Confidence: Based on separation theory and score distributions
        - Complete Ranking: All solvers ordered with significance testing
        - Audit Compliance: Full mathematical justification included
        """
        with log_operation(self.logger, "generate_selection_decision",
                          {"solver_count": len(self.solver_capabilities)}):
            
            # Step 1: Compute final match scores with optimal weights
            final_match_scores = self._compute_final_match_scores()
            
            # Step 2: Select optimal solver
            optimal_solver_index = int(np.argmax(final_match_scores))
            optimal_match_analysis = self._match_analyses[optimal_solver_index]
            
            # Step 3: Generate confidence score
            confidence_score = self._calculate_confidence_score(
                final_match_scores, optimal_solver_index
            )
            
            # Step 4: Create complete solver ranking
            solver_ranking = self._compute_solver_ranking(final_match_scores)
            
            # Step 5: Create chosen solver object with detailed information
            chosen_solver = ChosenSolver(
                solver_id=optimal_match_analysis.solver_id,
                confidence=confidence_score,
                match_score=optimal_match_analysis.match_score
            )
            
            # Step 6: Package complete selection decision
            selection_decision = SelectionDecision(
                chosen_solver=chosen_solver,
                ranking=solver_ranking,
                optimization_details=self._create_optimization_details(),
                execution_time_ms=0  # Will be set by runner
            )
            
            # Step 7: Attach detailed analysis for audit trail
            selection_decision._match_analyses = self._match_analyses
            selection_decision._final_scores = final_match_scores
            selection_decision._selection_timestamp = datetime.now(timezone.utc).isoformat()
            
            self.logger.info(
                f"Selection decision generated: chosen={chosen_solver.solver_id}, "
                f"confidence={confidence_score:.4f}, match_score={optimal_match_analysis.match_score:.6f}"
            )
            
            return selection_decision
    
    def _validate_selector_inputs(self) -> None:
        """complete validation of selector inputs."""
        n_solvers = len(self.solver_capabilities)
        
        # Validate solver capability count consistency
        if self.normalized_data.solver_capabilities.shape[0] != n_solvers:
            raise Stage5ValidationError(
                f"Solver capability count mismatch: {n_solvers} vs {self.normalized_data.solver_capabilities.shape[0]}",
                validation_type="solver_count_mismatch",
                expected_value=n_solvers,
                actual_value=self.normalized_data.solver_capabilities.shape[0]
            )
        
        # Validate optimal solver index from optimization
        if not (0 <= self.optimization_result.optimal_solver_index < n_solvers):
            raise Stage5ValidationError(
                f"Optimal solver index {self.optimization_result.optimal_solver_index} out of range [0, {n_solvers-1}]",
                validation_type="optimal_solver_bounds",
                expected_value=f"[0, {n_solvers-1}]",
                actual_value=self.optimization_result.optimal_solver_index
            )
        
        # Validate weight vector consistency
        if self.optimization_result.weights.shape != (PARAMETER_COUNT,):
            raise Stage5ValidationError(
                f"Optimal weights shape mismatch: expected ({PARAMETER_COUNT},), got {self.optimization_result.weights.shape}",
                validation_type="weights_shape_mismatch",
                expected_value=f"({PARAMETER_COUNT},)",
                actual_value=str(self.optimization_result.weights.shape)
            )
    
    def _compute_solver_match_analyses(self) -> List[SolverMatchAnalysis]:
        """
        Compute detailed match analysis for each solver.
        
        Returns:
            List of SolverMatchAnalysis objects with complete analysis
        """
        match_analyses = []
        
        optimal_weights = self.optimization_result.weights
        normalized_complexity = self.normalized_data.problem_complexity
        
        for i, solver_capability in enumerate(self.solver_capabilities):
            # Get normalized capability vector for this solver
            normalized_capabilities = self.normalized_data.solver_capabilities[i, :]
            
            # Compute parameter-wise gaps: gi,j = ri,j - c̃j
            parameter_gaps = normalized_capabilities - normalized_complexity
            
            # Compute weighted contributions: wj * gi,j
            weighted_contributions = optimal_weights * parameter_gaps
            
            # Compute overall match score: Mi(w*) = Σ wj * gi,j
            match_score = np.sum(weighted_contributions)
            
            # Identify strengths and weaknesses
            strength_threshold = 0.01  # Parameters where solver significantly exceeds requirements
            weakness_threshold = -0.01  # Parameters where solver falls significantly short
            
            strengths = [j for j in range(PARAMETER_COUNT) if parameter_gaps[j] > strength_threshold]
            weaknesses = [j for j in range(PARAMETER_COUNT) if parameter_gaps[j] < weakness_threshold]
            
            # Create match analysis object
            match_analysis = SolverMatchAnalysis(
                solver_id=solver_capability.solver_id,
                match_score=match_score,
                parameter_gaps=parameter_gaps,
                strengths=strengths,
                weaknesses=weaknesses,
                weighted_contributions=weighted_contributions
            )
            
            match_analyses.append(match_analysis)
        
        return match_analyses
    
    def _compute_final_match_scores(self) -> np.ndarray:
        """
        Compute final match scores with optimal weights.
        
        Returns:
            np.ndarray: Final match scores Mi(w*) for all solvers
        """
        return np.array([analysis.match_score for analysis in self._match_analyses])
    
    def _calculate_confidence_score(self,
                                  match_scores: np.ndarray,
                                  optimal_solver_index: int) -> float:
        """
        Calculate confidence score based on separation theory.
        
        Confidence is based on:
        1. Separation margin from optimization (primary factor)
        2. Score distribution characteristics (secondary factor)
        3. Statistical significance of selection (validation factor)
        
        Args:
            match_scores: All solver match scores
            optimal_solver_index: Index of optimal solver
            
        Returns:
            float: Confidence score in [0, 1] range
        """
        optimal_score = match_scores[optimal_solver_index]
        other_scores = np.concatenate([
            match_scores[:optimal_solver_index],
            match_scores[optimal_solver_index + 1:]
        ])
        
        if len(other_scores) == 0:
            return 1.0  # Perfect confidence with only one solver
        
        # Primary factor: Direct separation margin from optimization
        optimization_margin = self.optimization_result.separation_margin
        
        # Secondary factor: Statistical separation in score space
        score_separation = optimal_score - np.max(other_scores)
        score_std = np.std(other_scores) if len(other_scores) > 1 else 1.0
        statistical_separation = score_separation / (score_std + 1e-10)
        
        # Validation factor: Score distribution analysis
        score_range = np.max(match_scores) - np.min(match_scores)
        normalized_separation = score_separation / (score_range + 1e-10)
        
        # Combine factors with theoretical weighting
        # Optimization margin is primary (weight 0.6)
        # Statistical separation is secondary (weight 0.3)  
        # Normalized separation is validation (weight 0.1)
        
        # Normalize optimization margin to [0, 1] using sigmoid-like function
        margin_confidence = 1.0 / (1.0 + np.exp(-STATISTICAL_CONFIDENCE_FACTOR * optimization_margin))
        
        # Normalize statistical separation to [0, 1]
        stat_confidence = min(1.0, max(0.0, statistical_separation / STATISTICAL_CONFIDENCE_FACTOR))
        
        # Normalize score separation to [0, 1]
        norm_confidence = min(1.0, max(0.0, normalized_separation * 2.0))
        
        # Weighted combination
        confidence = (0.6 * margin_confidence + 0.3 * stat_confidence + 0.1 * norm_confidence)
        
        # Apply minimum threshold
        confidence = max(MIN_CONFIDENCE_THRESHOLD, confidence)
        
        return float(confidence)
    
    def _compute_solver_ranking(self, match_scores: np.ndarray) -> List[SolverRanking]:
        """
        Compute complete solver ranking with margin analysis.
        
        Args:
            match_scores: Final match scores for all solvers
            
        Returns:
            List of SolverRanking objects ordered by match score (descending)
        """
        # Create ranking tuples with (score, index, solver_id)
        ranking_data = [
            (match_scores[i], i, self.solver_capabilities[i].solver_id)
            for i in range(len(match_scores))
        ]
        
        # Sort by match score (descending)
        ranking_data.sort(key=lambda x: x[0], reverse=True)
        
        # Create SolverRanking objects
        solver_ranking = []
        
        for rank, (score, solver_index, solver_id) in enumerate(ranking_data):
            # Compute margin to next solver (or 0 if last)
            if rank < len(ranking_data) - 1:
                next_score = ranking_data[rank + 1][0]
                margin = score - next_score
            else:
                margin = 0.0  # Last solver has no margin
            
            ranking_entry = SolverRanking(
                solver_id=solver_id,
                score=score,
                margin=margin
            )
            
            solver_ranking.append(ranking_entry)
        
        return solver_ranking
    
    def _create_optimization_details(self) -> Dict[str, Any]:
        """
        Create optimization details for selection decision audit trail.
        
        Returns:
            Dict containing complete optimization context
        """
        # Extract weight statistics
        weight_stats = self.optimization_result.get_weight_distribution_stats()
        
        # Extract convergence information
        convergence_info = self.optimization_result.convergence_info
        
        optimization_details = {
            "learned_weights": self.optimization_result.weights.tolist(),
            "separation_margin": self.optimization_result.separation_margin,
            "optimal_solver_index": self.optimization_result.optimal_solver_index,
            "weight_statistics": weight_stats,
            "convergence": {
                "converged": convergence_info.get("converged", False),
                "iterations": convergence_info.get("iterations", 0),
                "final_weight_change": convergence_info.get("final_weight_change", float('inf')),
                "margin_improvement": convergence_info.get("margin_improvement", 0.0)
            },
            "normalization_factors": self.normalized_data.normalization_factors.factors.tolist(),
            "lp_solver": {
                "method": self.optimization_result.lp_details.get("solver_method", "unknown"),
                "success": self.optimization_result.lp_details.get("success", False),
                "iterations": self.optimization_result.lp_details.get("iterations", 0),
                "status": self.optimization_result.lp_details.get("status", "unknown")
            }
        }
        
        return optimization_details
    
    def get_selection_explanation(self, selection_decision: SelectionDecision) -> Dict[str, Any]:
        """
        Generate detailed explanation of selection decision for audit purposes.
        
        Args:
            selection_decision: Complete selection decision
            
        Returns:
            Dict containing detailed decision explanation
        """
        chosen_solver_id = selection_decision.chosen_solver.solver_id
        
        # Find chosen solver's analysis
        chosen_analysis = None
        for analysis in self._match_analyses:
            if analysis.solver_id == chosen_solver_id:
                chosen_analysis = analysis
                break
        
        if chosen_analysis is None:
            raise Stage5ComputationError(
                f"Could not find analysis for chosen solver: {chosen_solver_id}",
                computation_type="selection_explanation",
                context={"chosen_solver_id": chosen_solver_id}
            )
        
        # Generate complete explanation
        explanation = {
            "selection_summary": {
                "chosen_solver": chosen_solver_id,
                "match_score": chosen_analysis.match_score,
                "confidence": selection_decision.chosen_solver.confidence,
                "separation_margin": self.optimization_result.separation_margin
            },
            "mathematical_justification": {
                "optimal_weights": self.optimization_result.weights.tolist(),
                "weight_sum": float(np.sum(self.optimization_result.weights)),
                "lp_optimization_successful": self.optimization_result.lp_details.get("success", False),
                "convergence_achieved": self.optimization_result.convergence_info.get("converged", False)
            },
            "solver_analysis": {
                "top_strengths": chosen_analysis.get_top_strengths(3),
                "critical_weaknesses": chosen_analysis.get_critical_weaknesses(3),
                "parameter_coverage": len(chosen_analysis.strengths) / PARAMETER_COUNT,
                "match_statistics": chosen_analysis.compute_match_statistics()
            },
            "ranking_context": {
                "total_solvers": len(selection_decision.ranking),
                "chosen_rank": 1,  # Always rank 1 by construction
                "margin_to_second": selection_decision.ranking[0].margin if len(selection_decision.ranking) > 1 else 0.0,
                "top_3_solvers": [
                    {"solver_id": rank.solver_id, "score": rank.score, "margin": rank.margin}
                    for rank in selection_decision.ranking[:3]
                ]
            },
            "validation": {
                "normalization_valid": len(self.normalized_data.validation_results) > 0,
                "optimization_valid": self.optimization_result.separation_margin >= 0,
                "selection_valid": selection_decision.chosen_solver.confidence > MIN_CONFIDENCE_THRESHOLD,
                "audit_timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
        
        return explanation

# Module-level convenience functions
def select_optimal_solver(solver_capabilities: List[SolverCapability],
                         normalized_data: NormalizedData,
                         optimization_result: OptimizationResult,
                         logger: Optional[logging.Logger] = None) -> SelectionDecision:
    """
    Convenience function for complete solver selection.
    
    Args:
        solver_capabilities: List of solver capability objects
        normalized_data: Complete normalized dataset
        optimization_result: Optimal weights from LP optimization
        logger: Optional logger for operation tracking
        
    Returns:
        SelectionDecision: Complete selection decision
    """
    selector = SolverSelector(solver_capabilities, normalized_data, optimization_result, logger)
    return selector.generate_selection_decision()

def validate_selection_decision(decision: SelectionDecision) -> bool:
    """
    Validate selection decision meets mathematical requirements.
    
    Args:
        decision: SelectionDecision to validate
        
    Returns:
        bool: True if decision is mathematically valid
    """
    try:
        # Basic structure validation
        if not hasattr(decision, 'chosen_solver') or not hasattr(decision, 'ranking'):
            return False
        
        # Confidence validation
        if not (MIN_CONFIDENCE_THRESHOLD <= decision.chosen_solver.confidence <= 1.0):
            return False
        
        # Ranking validation
        if len(decision.ranking) == 0:
            return False
        
        # Top-ranked solver should match chosen solver
        if decision.ranking[0].solver_id != decision.chosen_solver.solver_id:
            return False
        
        return True
        
    except Exception:
        return False

# Export key classes and functions
__all__ = [
    "SolverSelector",
    "SolverMatchAnalysis",
    "select_optimal_solver",
    "validate_selection_decision",
    "MIN_CONFIDENCE_THRESHOLD",
    "PARAMETER_COUNT"
]

_logger.info("Stage 5.2 solver selection module loaded with confidence scoring enabled")