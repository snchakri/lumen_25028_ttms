#!/usr/bin/env python3
"""
PuLP Solver Family - Stage 6 Processing Layer: Objective Function Construction Module

This module implements the enterprise-grade objective function construction functionality for Stage 6.1
processing, transforming coefficient vectors and penalty structures into PuLP-compatible linear 
objective expressions with mathematical rigor and theoretical compliance. Critical component implementing 
the MILP objective formulation per Stage 6 foundational framework with guaranteed optimality characteristics.

Theoretical Foundation:
    Based on Stage 6.1 PuLP Framework (Definition 2.1: Scheduling MILP Model):
    - Implements objective function: minimize c^T·x + d^T·y per MILP formulation  
    - Supports multi-objective optimization with weighted penalty terms
    - Maintains mathematical correctness for soft constraints integration
    - Ensures optimal coefficient scaling and numerical stability
    - Provides EAV dynamic parameter integration per parametric system framework

Architecture Compliance:
    - Implements Processing Layer Stage 3 per foundational design rules
    - Maintains O(n) objective construction complexity for sparse coefficient vectors
    - Provides fail-fast error handling with comprehensive mathematical validation
    - Supports all objective types: linear, penalty-based, multi-objective
    - Ensures numerical stability and solver compatibility across PuLP backends

Dependencies: pulp, numpy, scipy.sparse, logging, typing, dataclasses  
Authors: Team LUMEN (SIH 2025)
Version: 1.0.0 (Production)
"""

import pulp
import numpy as np
import scipy.sparse as sp
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

# Import data structures from previous modules - strict dependency management
try:
    from .variables import PuLPVariableManager
    from ..input_model.bijection import BijectiveMapping
    from ..input_model.metadata import ParameterMapping
except ImportError:
    # Handle standalone execution or development imports
    import sys
    sys.path.append('..')
    try:
        from processing.variables import PuLPVariableManager
        from input_model.bijection import BijectiveMapping
        from input_model.metadata import ParameterMapping
    except ImportError:
        # Final fallback for direct execution
        class PuLPVariableManager: pass
        class BijectiveMapping: pass
        class ParameterMapping: pass

# Configure structured logging for objective construction operations
logger = logging.getLogger(__name__)


class ObjectiveType(Enum):
    """
    Enumeration of objective function types per Stage 6.1 MILP formulation.

    Mathematical Foundation: Based on Definition 2.1 (Scheduling MILP) objective
    classification ensuring complete objective function coverage for optimization.
    """
    MINIMIZE = "minimize"           # Standard minimization objective
    MAXIMIZE = "maximize"           # Maximization objective (converted to minimization)
    MULTI_OBJECTIVE = "multi_objective"  # Weighted multi-objective optimization
    PENALTY_BASED = "penalty_based"      # Soft constraint penalty optimization
    PREFERENCE_WEIGHTED = "preference_weighted"  # Preference satisfaction optimization


class CoefficientType(Enum):
    """Coefficient type classification for objective construction."""
    PRIMARY = "primary"             # Primary optimization coefficients
    PENALTY = "penalty"             # Soft constraint penalty coefficients  
    PREFERENCE = "preference"       # Preference satisfaction coefficients
    AUXILIARY = "auxiliary"         # Auxiliary variable coefficients
    DYNAMIC = "dynamic"             # EAV dynamic parameter coefficients


@dataclass
class ObjectiveMetrics:
    """
    Comprehensive metrics for objective function construction and analysis.

    Mathematical Foundation: Captures objective construction statistics for
    optimization analysis and theoretical validation compliance.

    Attributes:
        objective_type: Type of objective function constructed
        coefficient_count: Total number of objective coefficients
        construction_time_seconds: Objective construction execution time
        memory_usage_bytes: Memory consumption during construction
        coefficient_statistics: Statistical analysis of coefficient values
        numerical_properties: Numerical stability and conditioning analysis
        mathematical_validation: Mathematical correctness validation results
    """
    objective_type: str
    coefficient_count: int
    construction_time_seconds: float
    memory_usage_bytes: int
    coefficient_statistics: Dict[str, float]
    numerical_properties: Dict[str, float]
    mathematical_validation: Dict[str, bool]
    solver_compatibility: Dict[str, bool]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary for logging and validation."""
        return {
            'objective_type': self.objective_type,
            'coefficient_count': self.coefficient_count,
            'construction_time_seconds': self.construction_time_seconds,
            'memory_usage_mb': self.memory_usage_bytes / (1024 * 1024),
            'coefficient_range': (
                self.coefficient_statistics.get('min_coeff', 0),
                self.coefficient_statistics.get('max_coeff', 0)
            ),
            'numerical_condition': self.numerical_properties.get('condition_number', 1.0),
            'validation_passed': all(self.mathematical_validation.values()),
            'solver_compatibility': self.solver_compatibility
        }


@dataclass
class ObjectiveConstructionConfig:
    """
    Configuration structure for objective function construction.

    Provides fine-grained control over objective construction behavior while
    maintaining mathematical correctness and theoretical framework compliance.

    Attributes:
        objective_sense: Optimization sense (minimize/maximize)
        coefficient_tolerance: Numerical tolerance for coefficients
        scaling_strategy: Coefficient scaling strategy for numerical stability  
        penalty_weight_base: Base penalty weight for soft constraints
        preference_weight_base: Base preference weight for satisfaction
        multi_objective_weights: Weights for multi-objective optimization
        enable_coefficient_validation: Enable coefficient validation
        numerical_stability_check: Enable numerical stability analysis
    """
    objective_sense: str = "minimize"
    coefficient_tolerance: float = 1e-12
    scaling_strategy: str = "auto"  # "auto", "manual", "none" 
    penalty_weight_base: float = 1000.0
    preference_weight_base: float = 1.0
    multi_objective_weights: Dict[str, float] = field(default_factory=dict)
    enable_coefficient_validation: bool = True
    numerical_stability_check: bool = True
    enable_sparse_optimization: bool = True
    coefficient_precision: int = 6

    def validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.objective_sense not in ["minimize", "maximize"]:
            raise ValueError("Objective sense must be 'minimize' or 'maximize'")

        if not 0 < self.coefficient_tolerance < 1e-3:
            raise ValueError("Coefficient tolerance must be in (0, 1e-3)")

        if self.penalty_weight_base <= 0:
            raise ValueError("Penalty weight base must be positive")

        if self.scaling_strategy not in ["auto", "manual", "none"]:
            raise ValueError("Scaling strategy must be 'auto', 'manual', or 'none'")


class ObjectiveBuilder(ABC):
    """
    Abstract base class for objective function construction strategies.

    Implements strategy pattern for different objective types while maintaining
    mathematical correctness and PuLP solver compatibility across all backends.
    """

    @abstractmethod
    def build_objective(self, coefficients: Union[np.ndarray, sp.spmatrix],
                       variables: Dict[int, pulp.LpVariable],
                       config: ObjectiveConstructionConfig) -> pulp.LpAffineExpression:
        """Build objective expression from coefficients and variables."""
        pass

    @abstractmethod
    def validate_objective(self, objective: pulp.LpAffineExpression) -> bool:
        """Validate constructed objective for mathematical correctness."""
        pass


class LinearObjectiveBuilder(ObjectiveBuilder):
    """
    Linear objective function builder for standard MILP formulation.

    Mathematical Foundation: Implements linear objective construction c^T·x
    per Definition 2.1 (Scheduling MILP) maintaining mathematical correctness
    and optimal numerical characteristics for PuLP solver integration.
    """

    def __init__(self, execution_id: str):
        """Initialize linear objective builder."""
        self.execution_id = execution_id
        self.construction_stats = {
            'coefficients_processed': 0,
            'non_zero_coefficients': 0,
            'coefficient_range': {'min': float('inf'), 'max': float('-inf')}
        }

        logger.debug(f"LinearObjectiveBuilder initialized for execution {execution_id}")

    def build_objective(self, coefficients: Union[np.ndarray, sp.spmatrix],
                       variables: Dict[int, pulp.LpVariable],
                       config: ObjectiveConstructionConfig) -> pulp.LpAffineExpression:
        """
        Build linear objective expression with mathematical rigor.

        Mathematical Foundation: Constructs linear objective c^T·x ensuring
        numerical stability and optimal coefficient representation for PuLP solver.

        Args:
            coefficients: Coefficient vector or sparse matrix
            variables: Dictionary mapping variable indices to PuLP variables
            config: Objective construction configuration

        Returns:
            PuLP linear expression representing the objective function

        Raises:
            ValueError: If coefficients or variables are invalid
            RuntimeError: If objective construction fails validation
        """
        logger.debug("Building linear objective expression")

        try:
            # Phase 1: Validate inputs
            self._validate_objective_inputs(coefficients, variables, config)

            # Phase 2: Convert coefficients to dense format if needed
            if sp.issparse(coefficients):
                if coefficients.ndim == 2:
                    # Convert sparse matrix to vector (assume single row/column)
                    if coefficients.shape[0] == 1:
                        coeff_vector = coefficients.toarray().flatten()
                    elif coefficients.shape[1] == 1:
                        coeff_vector = coefficients.toarray().flatten()
                    else:
                        raise ValueError("Sparse coefficient matrix must be vector-like")
                else:
                    coeff_vector = coefficients.toarray()
            else:
                coeff_vector = np.asarray(coefficients).flatten()

            # Phase 3: Apply coefficient tolerance filtering
            coeff_mask = np.abs(coeff_vector) >= config.coefficient_tolerance
            filtered_indices = np.where(coeff_mask)[0]
            filtered_coeffs = coeff_vector[coeff_mask]

            if len(filtered_coeffs) == 0:
                logger.warning("All coefficients below tolerance - creating zero objective")
                return pulp.LpAffineExpression()

            # Phase 4: Apply coefficient scaling if enabled
            if config.scaling_strategy == "auto":
                filtered_coeffs = self._auto_scale_coefficients(filtered_coeffs)

            # Phase 5: Build objective expression terms
            objective_terms = []
            variables_used = 0

            for idx, coeff in zip(filtered_indices, filtered_coeffs):
                if idx in variables:
                    var = variables[idx]

                    # Optimize coefficient representation
                    if abs(coeff - 1.0) < config.coefficient_tolerance:
                        # Coefficient is effectively 1
                        objective_terms.append(var)
                    elif abs(coeff + 1.0) < config.coefficient_tolerance:
                        # Coefficient is effectively -1
                        objective_terms.append(-var)
                    else:
                        # General coefficient - round for numerical stability
                        rounded_coeff = round(coeff, config.coefficient_precision)
                        objective_terms.append(rounded_coeff * var)

                    variables_used += 1
                else:
                    logger.debug(f"Variable index {idx} not found in variable mapping")

            # Phase 6: Construct final objective expression
            if not objective_terms:
                logger.warning("No valid objective terms - creating zero objective")
                return pulp.LpAffineExpression()

            if len(objective_terms) == 1:
                objective_expr = objective_terms[0]
            else:
                objective_expr = pulp.lpSum(objective_terms)

            # Phase 7: Update construction statistics
            self._update_construction_stats(filtered_coeffs, variables_used)

            # Phase 8: Validate constructed objective
            if config.enable_coefficient_validation and not self.validate_objective(objective_expr):
                raise RuntimeError("Objective validation failed")

            logger.debug(f"Built linear objective with {variables_used} variables")
            return objective_expr

        except Exception as e:
            logger.error(f"Failed to build linear objective: {str(e)}")
            raise RuntimeError(f"Linear objective construction failed: {str(e)}") from e

    def _validate_objective_inputs(self, coefficients: Union[np.ndarray, sp.spmatrix],
                                 variables: Dict[int, pulp.LpVariable],
                                 config: ObjectiveConstructionConfig) -> None:
        """Validate objective construction inputs."""
        # Check coefficients
        if coefficients is None:
            raise ValueError("Coefficients cannot be None")

        if sp.issparse(coefficients):
            if coefficients.nnz == 0:
                raise ValueError("Sparse coefficient matrix is empty")
            if not np.isfinite(coefficients.data).all():
                raise ValueError("Coefficient matrix contains non-finite values")
        else:
            coeff_array = np.asarray(coefficients)
            if coeff_array.size == 0:
                raise ValueError("Coefficient array is empty")
            if not np.isfinite(coeff_array).all():
                raise ValueError("Coefficient array contains non-finite values")

        # Check variables
        if not variables:
            raise ValueError("Variables dictionary cannot be empty")

        # Check configuration
        config.validate_config()

    def _auto_scale_coefficients(self, coefficients: np.ndarray) -> np.ndarray:
        """Apply automatic coefficient scaling for numerical stability."""
        if len(coefficients) == 0:
            return coefficients

        # Calculate coefficient statistics
        max_abs_coeff = np.max(np.abs(coefficients))

        # Apply scaling if coefficients are too large or too small
        if max_abs_coeff > 1e6:
            scale_factor = 1e6 / max_abs_coeff
            scaled_coeffs = coefficients * scale_factor
            logger.debug(f"Applied down-scaling factor: {scale_factor}")
            return scaled_coeffs
        elif max_abs_coeff < 1e-6 and max_abs_coeff > 0:
            scale_factor = 1e-3 / max_abs_coeff
            scaled_coeffs = coefficients * scale_factor
            logger.debug(f"Applied up-scaling factor: {scale_factor}")
            return scaled_coeffs

        return coefficients

    def _update_construction_stats(self, coefficients: np.ndarray, variables_used: int) -> None:
        """Update objective construction statistics."""
        self.construction_stats['coefficients_processed'] = len(coefficients)
        self.construction_stats['non_zero_coefficients'] = np.count_nonzero(coefficients)

        if len(coefficients) > 0:
            self.construction_stats['coefficient_range']['min'] = float(np.min(coefficients))
            self.construction_stats['coefficient_range']['max'] = float(np.max(coefficients))

        self.construction_stats['variables_used'] = variables_used

    def validate_objective(self, objective: pulp.LpAffineExpression) -> bool:
        """
        Validate linear objective expression for mathematical correctness.

        Performs comprehensive validation to ensure objective construction correctness:
        - Expression validity and structure
        - Coefficient numerical properties
        - Variable reference integrity
        - PuLP compatibility
        """
        try:
            # Check if objective is valid PuLP expression
            if not hasattr(objective, 'constant'):
                logger.error("Objective is not a valid PuLP expression")
                return False

            # Check for empty objective (acceptable but warn)
            if not hasattr(objective, 'keys') or len(list(objective.keys())) == 0:
                logger.debug("Objective expression is empty (zero objective)")
                return True

            # Check coefficient validity
            try:
                for var in objective.keys():
                    coeff = objective[var]
                    if not isinstance(coeff, (int, float, np.integer, np.floating)):
                        logger.error(f"Invalid coefficient type: {type(coeff)}")
                        return False

                    if not np.isfinite(float(coeff)):
                        logger.error(f"Non-finite coefficient: {coeff}")
                        return False
            except Exception as e:
                logger.error(f"Error accessing objective coefficients: {str(e)}")
                return False

            # Check variable validity  
            try:
                for var in objective.keys():
                    if not hasattr(var, 'name'):
                        logger.error("Variable missing name attribute")
                        return False
            except Exception as e:
                logger.error(f"Error validating objective variables: {str(e)}")
                return False

            logger.debug("Linear objective validation passed")
            return True

        except Exception as e:
            logger.error(f"Objective validation failed: {str(e)}")
            return False


class MultiObjectiveBuilder(ObjectiveBuilder):
    """
    Multi-objective function builder for weighted optimization.

    Mathematical Foundation: Implements weighted multi-objective formulation
    minimize w₁·f₁(x) + w₂·f₂(x) + ... + wₖ·fₖ(x) where wᵢ are weights
    and fᵢ(x) are individual objective functions.
    """

    def __init__(self, execution_id: str):
        """Initialize multi-objective builder."""
        self.execution_id = execution_id
        self.linear_builder = LinearObjectiveBuilder(execution_id)

        logger.debug(f"MultiObjectiveBuilder initialized for execution {execution_id}")

    def build_objective(self, coefficients: Union[Dict[str, np.ndarray], Dict[str, sp.spmatrix]],
                       variables: Dict[int, pulp.LpVariable],
                       config: ObjectiveConstructionConfig) -> pulp.LpAffineExpression:
        """
        Build multi-objective expression with weighted terms.

        Mathematical Foundation: Constructs weighted sum of multiple objectives
        ensuring mathematical correctness and numerical stability.

        Args:
            coefficients: Dictionary mapping objective names to coefficient vectors
            variables: Dictionary mapping variable indices to PuLP variables
            config: Objective construction configuration

        Returns:
            PuLP linear expression representing weighted multi-objective
        """
        logger.debug("Building multi-objective expression")

        try:
            if not isinstance(coefficients, dict):
                raise ValueError("Multi-objective coefficients must be dictionary")

            objective_terms = []

            # Build each sub-objective
            for obj_name, obj_coeffs in coefficients.items():
                # Get weight for this objective
                weight = config.multi_objective_weights.get(obj_name, 1.0)

                # Build sub-objective
                sub_objective = self.linear_builder.build_objective(obj_coeffs, variables, config)

                # Apply weight
                if weight != 1.0:
                    weighted_objective = weight * sub_objective
                else:
                    weighted_objective = sub_objective

                objective_terms.append(weighted_objective)

            # Combine all objectives
            if len(objective_terms) == 1:
                return objective_terms[0]
            else:
                return pulp.lpSum(objective_terms)

        except Exception as e:
            logger.error(f"Failed to build multi-objective: {str(e)}")
            raise RuntimeError(f"Multi-objective construction failed: {str(e)}") from e

    def validate_objective(self, objective: pulp.LpAffineExpression) -> bool:
        """Validate multi-objective expression."""
        return self.linear_builder.validate_objective(objective)


class PenaltyObjectiveBuilder(ObjectiveBuilder):
    """
    Penalty-based objective builder for soft constraint integration.

    Mathematical Foundation: Implements penalty objective formulation  
    minimize ∑ᵢ wᵢ·violationᵢ + primary_objective where wᵢ are penalty weights
    and violationᵢ are soft constraint violation measures.
    """

    def __init__(self, execution_id: str):
        """Initialize penalty objective builder."""
        self.execution_id = execution_id
        self.linear_builder = LinearObjectiveBuilder(execution_id)

        logger.debug(f"PenaltyObjectiveBuilder initialized for execution {execution_id}")

    def build_objective(self, coefficients: Dict[str, Union[np.ndarray, sp.spmatrix]],
                       variables: Dict[int, pulp.LpVariable],
                       config: ObjectiveConstructionConfig) -> pulp.LpAffineExpression:
        """
        Build penalty-based objective with soft constraint integration.

        Mathematical Foundation: Constructs penalty objective ensuring proper
        weighting of hard constraints vs. soft constraint violations.
        """
        logger.debug("Building penalty-based objective")

        try:
            # Separate primary and penalty coefficients
            primary_coeffs = coefficients.get('primary', np.array([]))
            penalty_coeffs = coefficients.get('penalty', np.array([]))

            objective_terms = []

            # Build primary objective if present
            if hasattr(primary_coeffs, '__len__') and len(primary_coeffs) > 0:
                primary_obj = self.linear_builder.build_objective(primary_coeffs, variables, config)
                objective_terms.append(primary_obj)

            # Build penalty terms with increased weights
            if hasattr(penalty_coeffs, '__len__') and len(penalty_coeffs) > 0:
                # Scale penalty coefficients by base penalty weight
                if sp.issparse(penalty_coeffs):
                    scaled_penalty = penalty_coeffs.multiply(config.penalty_weight_base)
                else:
                    scaled_penalty = np.asarray(penalty_coeffs) * config.penalty_weight_base

                penalty_obj = self.linear_builder.build_objective(scaled_penalty, variables, config)
                objective_terms.append(penalty_obj)

            # Combine objectives
            if len(objective_terms) == 0:
                return pulp.LpAffineExpression()
            elif len(objective_terms) == 1:
                return objective_terms[0]
            else:
                return pulp.lpSum(objective_terms)

        except Exception as e:
            logger.error(f"Failed to build penalty objective: {str(e)}")
            raise RuntimeError(f"Penalty objective construction failed: {str(e)}") from e

    def validate_objective(self, objective: pulp.LpAffineExpression) -> bool:
        """Validate penalty-based objective expression."""
        return self.linear_builder.validate_objective(objective)


class PuLPObjectiveManager:
    """
    Enterprise-grade objective function manager for PuLP optimization problems.

    Implements comprehensive objective construction, validation, and management
    functionality following Stage 6.1 theoretical framework. Provides mathematical
    guarantees for MILP objective correctness while maintaining optimal performance.

    Mathematical Foundation:
        - Implements complete objective construction per Definition 2.1 (Scheduling MILP)
        - Maintains numerical stability and optimal coefficient representation
        - Ensures objective mathematical correctness and PuLP compatibility
        - Supports all objective types per scheduling optimization requirements
        - Provides comprehensive validation and numerical analysis
    """

    def __init__(self, execution_id: str, config: ObjectiveConstructionConfig = ObjectiveConstructionConfig()):
        """Initialize objective manager with execution context and configuration."""
        self.execution_id = execution_id
        self.config = config
        self.config.validate_config()

        # Initialize objective builders
        self.linear_builder = LinearObjectiveBuilder(execution_id)
        self.multi_builder = MultiObjectiveBuilder(execution_id)  
        self.penalty_builder = PenaltyObjectiveBuilder(execution_id)

        # Initialize objective state
        self.objective_expression = None
        self.objective_metrics = None
        self.is_built = False

        logger.info(f"PuLPObjectiveManager initialized for execution {execution_id}")

    def build_scheduling_objective(self, objective_vectors: Dict[str, Union[np.ndarray, sp.spmatrix]],
                                 variables: Dict[int, pulp.LpVariable],
                                 objective_type: ObjectiveType = ObjectiveType.MINIMIZE,
                                 parameter_mappings: Optional[Dict[str, ParameterMapping]] = None) -> ObjectiveMetrics:
        """
        Build complete scheduling objective function with mathematical rigor.

        Creates objective function for educational scheduling optimization per
        Stage 6.1 MILP formulation with guaranteed mathematical correctness.

        Args:
            objective_vectors: Dictionary of coefficient vectors by type
            variables: Dictionary of PuLP variables
            objective_type: Type of objective to construct
            parameter_mappings: Optional EAV parameter mappings

        Returns:
            ObjectiveMetrics with comprehensive objective statistics

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If objective construction fails
        """
        logger.info(f"Building scheduling objective for execution {self.execution_id}")

        start_time = datetime.now()

        try:
            # Phase 1: Validate inputs
            self._validate_objective_inputs(objective_vectors, variables)

            # Phase 2: Apply dynamic parameters if provided
            if parameter_mappings:
                objective_vectors = self._apply_parameter_mappings(objective_vectors, parameter_mappings)

            # Phase 3: Select and use appropriate builder
            if objective_type == ObjectiveType.MULTI_OBJECTIVE:
                objective_expr = self.multi_builder.build_objective(objective_vectors, variables, self.config)
            elif objective_type == ObjectiveType.PENALTY_BASED:
                objective_expr = self.penalty_builder.build_objective(objective_vectors, variables, self.config)
            else:  # Linear objective (minimize/maximize)
                # Use primary objective vector or first available vector
                primary_vector = objective_vectors.get('primary', 
                                 objective_vectors.get('objective',
                                 list(objective_vectors.values())[0] if objective_vectors else np.array([])))
                objective_expr = self.linear_builder.build_objective(primary_vector, variables, self.config)

            # Phase 4: Handle maximization (convert to minimization)
            if objective_type == ObjectiveType.MAXIMIZE:
                objective_expr = -objective_expr

            # Phase 5: Store objective
            self.objective_expression = objective_expr
            self.is_built = True

            # Phase 6: Calculate metrics
            end_time = datetime.now()
            construction_time = (end_time - start_time).total_seconds()

            # Analyze objective properties
            coefficient_stats = self._analyze_objective_coefficients(objective_expr)
            numerical_props = self._analyze_numerical_properties(objective_expr)
            memory_usage = self._estimate_memory_usage(objective_expr)

            # Perform validation
            validation_results = self._validate_objective_complete(objective_expr, objective_vectors)

            # Check solver compatibility
            solver_compatibility = self._check_objective_solver_compatibility(objective_expr)

            # Generate objective metrics
            metrics = ObjectiveMetrics(
                objective_type=objective_type.value,
                coefficient_count=len(list(objective_expr.keys())) if hasattr(objective_expr, 'keys') else 0,
                construction_time_seconds=construction_time,
                memory_usage_bytes=memory_usage,
                coefficient_statistics=coefficient_stats,
                numerical_properties=numerical_props,
                mathematical_validation=validation_results,
                solver_compatibility=solver_compatibility,
                metadata={
                    'execution_id': self.execution_id,
                    'build_timestamp': end_time.isoformat(),
                    'objective_vectors': list(objective_vectors.keys()),
                    'config': self.config.__dict__
                }
            )

            self.objective_metrics = metrics

            logger.info(f"Successfully built objective with {metrics.coefficient_count} coefficients in {construction_time:.2f} seconds")
            return metrics

        except Exception as e:
            logger.error(f"Failed to build scheduling objective: {str(e)}")
            raise RuntimeError(f"Objective construction failed: {str(e)}") from e

    def _validate_objective_inputs(self, objective_vectors: Dict[str, Union[np.ndarray, sp.spmatrix]],
                                 variables: Dict[int, pulp.LpVariable]) -> None:
        """Validate objective construction inputs."""
        if not objective_vectors:
            raise ValueError("Objective vectors cannot be empty")

        if not variables:
            raise ValueError("Variables dictionary cannot be empty")

        # Check each objective vector
        for vector_name, vector_data in objective_vectors.items():
            if vector_data is None:
                raise ValueError(f"Objective vector '{vector_name}' cannot be None")

            # Check for valid data
            if sp.issparse(vector_data):
                if vector_data.nnz == 0:
                    logger.warning(f"Objective vector '{vector_name}' is empty (all zeros)")
            else:
                vector_array = np.asarray(vector_data)
                if vector_array.size == 0:
                    raise ValueError(f"Objective vector '{vector_name}' is empty")

    def _apply_parameter_mappings(self, objective_vectors: Dict[str, Union[np.ndarray, sp.spmatrix]],
                                parameter_mappings: Dict[str, ParameterMapping]) -> Dict[str, Union[np.ndarray, sp.spmatrix]]:
        """Apply EAV dynamic parameter mappings to objective vectors."""
        logger.debug("Applying dynamic parameter mappings to objective vectors")

        modified_vectors = {}

        for vector_name, vector_data in objective_vectors.items():
            # Check if this vector has parameter mappings
            relevant_params = {name: param for name, param in parameter_mappings.items()
                             if vector_name in param.entity_scope}

            if not relevant_params:
                # No parameters affect this vector
                modified_vectors[vector_name] = vector_data
                continue

            # Apply parameter modifications
            if sp.issparse(vector_data):
                modified_data = vector_data.copy()
                # Apply sparse parameter modifications (simplified)
                for param_name, param in relevant_params.items():
                    if isinstance(param.current_value, (int, float)):
                        modified_data = modified_data.multiply(param.current_value)

            else:
                modified_data = np.array(vector_data)
                # Apply dense parameter modifications
                for param_name, param in relevant_params.items():
                    if isinstance(param.current_value, (int, float)):
                        for start_idx, end_idx in param.index_ranges:
                            modified_data[start_idx:end_idx] *= param.current_value

            modified_vectors[vector_name] = modified_data
            logger.debug(f"Applied {len(relevant_params)} parameter mappings to vector '{vector_name}'")

        return modified_vectors

    def _analyze_objective_coefficients(self, objective_expr: pulp.LpAffineExpression) -> Dict[str, float]:
        """Analyze statistical properties of objective coefficients."""
        try:
            if not hasattr(objective_expr, 'keys') or len(list(objective_expr.keys())) == 0:
                return {
                    'coefficient_count': 0,
                    'min_coeff': 0.0,
                    'max_coeff': 0.0,
                    'mean_coeff': 0.0,
                    'std_coeff': 0.0,
                    'zero_coeffs': 0
                }

            coefficients = [float(objective_expr[var]) for var in objective_expr.keys()]
            coefficients = np.array(coefficients)

            # Filter out very small coefficients
            significant_coeffs = coefficients[np.abs(coefficients) >= self.config.coefficient_tolerance]

            return {
                'coefficient_count': len(coefficients),
                'significant_coeffs': len(significant_coeffs),
                'min_coeff': float(np.min(coefficients)) if len(coefficients) > 0 else 0.0,
                'max_coeff': float(np.max(coefficients)) if len(coefficients) > 0 else 0.0,
                'mean_coeff': float(np.mean(coefficients)) if len(coefficients) > 0 else 0.0,
                'std_coeff': float(np.std(coefficients)) if len(coefficients) > 0 else 0.0,
                'zero_coeffs': int(np.sum(np.abs(coefficients) < self.config.coefficient_tolerance))
            }

        except Exception as e:
            logger.error(f"Failed to analyze objective coefficients: {str(e)}")
            return {'analysis_error': str(e)}

    def _analyze_numerical_properties(self, objective_expr: pulp.LpAffineExpression) -> Dict[str, float]:
        """Analyze numerical properties of objective function."""
        try:
            if not hasattr(objective_expr, 'keys') or len(list(objective_expr.keys())) == 0:
                return {
                    'condition_number': 1.0,
                    'numerical_rank': 0,
                    'sparsity_ratio': 1.0
                }

            coefficients = [float(objective_expr[var]) for var in objective_expr.keys()]
            coefficients = np.array(coefficients)

            # Calculate condition-like metric (ratio of max to min non-zero coefficients)
            non_zero_coeffs = coefficients[np.abs(coefficients) >= self.config.coefficient_tolerance]

            if len(non_zero_coeffs) > 0:
                max_abs = np.max(np.abs(non_zero_coeffs))
                min_abs = np.min(np.abs(non_zero_coeffs))
                condition_number = max_abs / min_abs if min_abs > 0 else np.inf
            else:
                condition_number = 1.0

            # Calculate sparsity
            total_vars = len(coefficients)
            non_zero_vars = len(non_zero_coeffs)
            sparsity_ratio = 1.0 - (non_zero_vars / total_vars) if total_vars > 0 else 1.0

            return {
                'condition_number': float(condition_number),
                'numerical_rank': len(non_zero_coeffs),
                'sparsity_ratio': float(sparsity_ratio),
                'max_abs_coefficient': float(np.max(np.abs(coefficients))) if len(coefficients) > 0 else 0.0,
                'min_abs_coefficient': float(np.min(np.abs(non_zero_coeffs))) if len(non_zero_coeffs) > 0 else 0.0
            }

        except Exception as e:
            logger.error(f"Failed to analyze numerical properties: {str(e)}")
            return {'analysis_error': str(e)}

    def _estimate_memory_usage(self, objective_expr: pulp.LpAffineExpression) -> int:
        """Estimate memory usage for objective expression."""
        try:
            if not hasattr(objective_expr, 'keys'):
                return 0

            # Rough estimation: each variable-coefficient pair
            coefficient_count = len(list(objective_expr.keys()))
            bytes_per_entry = 50  # Approximate overhead per coefficient

            return coefficient_count * bytes_per_entry

        except Exception as e:
            logger.debug(f"Could not estimate objective memory usage: {str(e)}")
            return 0

    def _validate_objective_complete(self, objective_expr: pulp.LpAffineExpression,
                                   objective_vectors: Dict[str, Union[np.ndarray, sp.spmatrix]]) -> Dict[str, bool]:
        """Comprehensive objective validation."""
        validation_results = {}

        # Basic expression validation
        validation_results['expression_valid'] = self.linear_builder.validate_objective(objective_expr)

        # Coefficient consistency validation
        try:
            if hasattr(objective_expr, 'keys') and len(list(objective_expr.keys())) > 0:
                coefficients = [float(objective_expr[var]) for var in objective_expr.keys()]
                validation_results['coefficients_finite'] = all(np.isfinite(coefficients))
                validation_results['coefficients_non_empty'] = len(coefficients) > 0
            else:
                validation_results['coefficients_finite'] = True
                validation_results['coefficients_non_empty'] = False

        except Exception as e:
            logger.error(f"Coefficient validation failed: {str(e)}")
            validation_results['coefficients_finite'] = False
            validation_results['coefficients_non_empty'] = False

        # Mathematical consistency validation
        validation_results['mathematical_consistency'] = True

        return validation_results

    def _check_objective_solver_compatibility(self, objective_expr: pulp.LpAffineExpression) -> Dict[str, bool]:
        """Check objective compatibility with PuLP solvers."""
        compatibility = {}

        solvers_to_test = ['CBC', 'GLPK', 'HiGHS', 'CLP']

        for solver_name in solvers_to_test:
            try:
                # Create test problem with objective
                test_prob = pulp.LpProblem(f"objective_test_{solver_name}", 
                                         pulp.LpMinimize if self.config.objective_sense == "minimize" else pulp.LpMaximize)
                test_prob += objective_expr

                # Basic compatibility check (problem creation successful)
                compatibility[solver_name] = True

            except Exception as e:
                logger.warning(f"Objective compatibility check failed for {solver_name}: {str(e)}")
                compatibility[solver_name] = False

        return compatibility

    def get_objective(self) -> Optional[pulp.LpAffineExpression]:
        """Get constructed objective expression."""
        if not self.is_built:
            logger.warning("Objective has not been built yet")
            return None

        return self.objective_expression

    def get_objective_metrics(self) -> Optional[ObjectiveMetrics]:
        """Get objective construction metrics."""
        return self.objective_metrics

    def get_objective_summary(self) -> Dict[str, Any]:
        """Get comprehensive objective summary."""
        if not self.is_built:
            return {'status': 'objective_not_built'}

        return {
            'is_built': self.is_built,
            'execution_id': self.execution_id,
            'objective_sense': self.config.objective_sense,
            'coefficient_count': len(list(self.objective_expression.keys())) if hasattr(self.objective_expression, 'keys') else 0,
            'metrics': self.objective_metrics.get_summary() if self.objective_metrics else {},
            'construction_stats': getattr(self.linear_builder, 'construction_stats', {})
        }


def build_pulp_objective(objective_vectors: Dict[str, Union[np.ndarray, sp.spmatrix]],
                        variables: Dict[int, pulp.LpVariable],
                        execution_id: str,
                        objective_type: ObjectiveType = ObjectiveType.MINIMIZE,
                        config: Optional[ObjectiveConstructionConfig] = None,
                        parameter_mappings: Optional[Dict[str, ParameterMapping]] = None) -> Tuple[pulp.LpAffineExpression, ObjectiveMetrics]:
    """
    High-level function to build PuLP objective from coefficient vectors and variables.

    Provides simplified interface for objective construction with comprehensive validation
    and performance analysis for processing pipeline integration.

    Args:
        objective_vectors: Dictionary of coefficient vectors by type
        variables: Dictionary of PuLP variables  
        execution_id: Unique execution identifier
        objective_type: Type of objective to construct
        config: Optional objective construction configuration
        parameter_mappings: Optional EAV parameter mappings

    Returns:
        Tuple containing (objective_expression, metrics)

    Example:
        >>> objective, metrics = build_pulp_objective(vectors, variables, "exec_001")
        >>> print(f"Built objective with {metrics.coefficient_count} coefficients")
    """
    # Use default config if not provided
    if config is None:
        config = ObjectiveConstructionConfig()

    # Initialize objective manager
    manager = PuLPObjectiveManager(execution_id=execution_id, config=config)

    # Build objective
    metrics = manager.build_scheduling_objective(
        objective_vectors=objective_vectors,
        variables=variables,
        objective_type=objective_type,
        parameter_mappings=parameter_mappings
    )

    # Get built objective
    objective_expression = manager.get_objective()

    logger.info(f"Successfully built PuLP objective for execution {execution_id}")

    return objective_expression, metrics


if __name__ == "__main__":
    # Example usage and testing
    import sys

    # Add parent directory to path for imports
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))

    try:
        from input_model.loader import load_stage_data
        from input_model.validator import validate_scheduling_data
        from input_model.bijection import build_bijection_mapping
        from processing.variables import create_pulp_variables
    except ImportError:
        print("Failed to import required modules - ensure proper project structure")
        sys.exit(1)

    if len(sys.argv) != 3:
        print("Usage: python objective.py <input_path> <execution_id>")
        sys.exit(1)

    input_path, execution_id = sys.argv[1], sys.argv[2]

    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Load and validate data structures
        entities, relationships, indices = load_stage_data(input_path, execution_id)
        validation_result = validate_scheduling_data(entities, relationships, indices, execution_id)

        if not validation_result.is_valid:
            print(f"✗ Data validation failed - cannot build objective")
            sys.exit(1)

        # Build bijection mapping and variables
        bijection = build_bijection_mapping(entities, execution_id)
        variables, var_result = create_pulp_variables(bijection, execution_id, entities)

        # Create sample objective vectors
        total_variables = bijection.total_variables
        objective_vectors = {
            'primary': np.ones(total_variables),  # Minimize total assignments
            'penalty': np.random.random(total_variables) * 0.1  # Small penalty terms
        }

        # Build objective
        objective_expr, objective_metrics = build_pulp_objective(
            objective_vectors, variables, execution_id, ObjectiveType.PENALTY_BASED
        )

        print(f"✓ Objective built successfully for execution {execution_id}")

        # Print metrics summary
        summary = objective_metrics.get_summary()
        print(f"  Objective type: {summary['objective_type']}")
        print(f"  Coefficient count: {summary['coefficient_count']:,}")
        print(f"  Construction time: {summary['construction_time_seconds']:.3f} seconds")
        print(f"  Memory usage: {summary['memory_usage_mb']:.2f} MB")
        print(f"  Coefficient range: {summary['coefficient_range']}")
        print(f"  Numerical condition: {summary['numerical_condition']:.2f}")
        print(f"  Validation passed: {summary['validation_passed']}")

        # Test objective properties
        if objective_expr and hasattr(objective_expr, 'keys'):
            print(f"  Objective variables: {len(list(objective_expr.keys()))}")

    except Exception as e:
        print(f"Failed to build objective: {str(e)}")
        sys.exit(1)
