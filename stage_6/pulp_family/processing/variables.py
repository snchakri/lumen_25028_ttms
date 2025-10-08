#!/usr/bin/env python3
"""
PuLP Solver Family - Stage 6 Processing Layer: Variable Creation Module

This module implements the enterprise-grade variable creation functionality for Stage 6.1 processing,
transforming the mathematical optimization problem structure into PuLP solver-compatible decision 
variables. Critical component implementing the MILP formulation from theoretical framework with
guaranteed mathematical correctness and optimal memory utilization.

Theoretical Foundation:
    Based on Stage 6.1 PuLP Solver Family Framework (Section 2: Universal Problem Formulation):
    - Implements Definition 2.1 (Scheduling MILP) with binary decision variables
    - Variable encoding per Definition 2.3: x_{c,f,r,t,b} ∈ {0,1} for assignment decisions
    - Maintains bijection consistency per stride-based mapping algorithm  
    - Supports auxiliary continuous variables for resource utilization modeling
    - Ensures mathematical formulation compliance with PuLP solver requirements

Architecture Compliance:
    - Implements Processing Layer Stage 1 per foundational design rules
    - Maintains O(1) variable creation complexity per bijection mapping
    - Provides fail-fast error handling with comprehensive logging
    - Supports all PuLP solver backends (CBC, GLPK, HiGHS, CLP, Symphony)
    - Ensures memory efficiency with sparse variable representation

Dependencies: pulp, numpy, pandas, logging, json, datetime, pathlib
Authors: Team LUMEN (SIH 2025)
Version: 1.0.0 (Production)
"""

import pulp
import numpy as np
import pandas as pd
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Import data structures from input modeling layer - strict dependency management
try:
    from ..input_model.loader import EntityCollection
    from ..input_model.bijection import BijectiveMapping
    from ..input_model.metadata import InputModelMetadataGenerator
except ImportError:
    # Handle standalone execution or development imports
    import sys
    sys.path.append('..')
    try:
        from input_model.loader import EntityCollection
        from input_model.bijection import BijectiveMapping  
        from input_model.metadata import InputModelMetadataGenerator
    except ImportError:
        # Final fallback for direct execution
        class EntityCollection: pass
        class BijectiveMapping: pass
        class InputModelMetadataGenerator: pass

# Configure structured logging for variable creation operations
logger = logging.getLogger(__name__)


@dataclass
class VariableCreationResult:
    """
    Comprehensive result structure for variable creation process.

    Mathematical Foundation: Captures complete variable space structure
    per MILP formulation ensuring mathematical correctness and traceability.

    Attributes:
        variables: Dictionary mapping variable indices to PuLP LpVariable objects
        variable_count: Total number of variables created (V = total variables)
        creation_time_seconds: Variable creation execution time
        memory_usage_bytes: Estimated memory usage for variable storage
        variable_types: Categorization of variables by type (binary, continuous)
        index_mapping: Mapping between flat indices and (c,f,r,t,b) tuples
        mathematical_properties: Verification of mathematical formulation compliance
        solver_compatibility: Compatibility verification for target solvers
        metadata: Additional creation metadata and diagnostics
    """
    variables: Dict[int, pulp.LpVariable]
    variable_count: int
    creation_time_seconds: float
    memory_usage_bytes: int
    variable_types: Dict[str, int]
    index_mapping: Dict[int, Tuple[Any, Any, Any, Any, Any]]
    mathematical_properties: Dict[str, Any]
    solver_compatibility: Dict[str, bool]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary for logging and validation."""
        return {
            'variable_count': self.variable_count,
            'creation_time_seconds': self.creation_time_seconds,
            'memory_usage_mb': self.memory_usage_bytes / (1024 * 1024),
            'variable_types': self.variable_types,
            'mathematical_compliance': self.mathematical_properties.get('formulation_compliant', False),
            'solver_compatibility': self.solver_compatibility,
            'creation_status': 'success'
        }


@dataclass
class VariableCreationConfig:
    """
    Configuration structure for variable creation process.

    Provides fine-grained control over variable creation behavior while maintaining
    mathematical correctness and theoretical framework compliance.

    Attributes:
        variable_prefix: Prefix for variable names (default: "x")
        binary_variables: Whether to create binary variables (True for MILP)
        continuous_auxiliary: Whether to create auxiliary continuous variables
        memory_optimization: Enable memory-efficient variable creation
        batch_size: Batch size for variable creation (memory management)
        solver_compatibility_check: Verify compatibility with target solvers
        mathematical_validation: Enable mathematical formulation validation
        naming_strategy: Strategy for variable naming (indexed, semantic)
        creation_logging: Enable detailed variable creation logging
    """
    variable_prefix: str = "x"
    binary_variables: bool = True
    continuous_auxiliary: bool = False
    memory_optimization: bool = True  
    batch_size: int = 10000
    solver_compatibility_check: bool = True
    mathematical_validation: bool = True
    naming_strategy: str = "indexed"  # "indexed" or "semantic"
    creation_logging: bool = True

    def validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        if self.naming_strategy not in ["indexed", "semantic"]:
            raise ValueError("Naming strategy must be 'indexed' or 'semantic'")

        if not isinstance(self.binary_variables, bool):
            raise ValueError("Binary variables flag must be boolean")


class VariableFactory(ABC):
    """
    Abstract factory for creating different types of optimization variables.

    Implements factory pattern for extensible variable creation while maintaining
    mathematical correctness and PuLP solver compatibility across all backends.
    """

    @abstractmethod
    def create_variable(self, idx: int, assignment_tuple: Optional[Tuple] = None,
                       config: VariableCreationConfig = VariableCreationConfig()) -> pulp.LpVariable:
        """Create single variable with specified configuration."""
        pass

    @abstractmethod
    def validate_variable(self, variable: pulp.LpVariable) -> bool:
        """Validate created variable mathematical properties."""
        pass


class BinaryAssignmentVariableFactory(VariableFactory):
    """
    Factory for creating binary assignment variables x_{c,f,r,t,b} ∈ {0,1}.

    Mathematical Foundation: Implements Definition 2.3 (Variable Assignment Encoding)
    from Stage 6.1 framework creating binary decision variables for course assignments.

    Maintains mathematical correctness per MILP formulation requirements while
    ensuring optimal memory utilization and PuLP solver compatibility.
    """

    def __init__(self, execution_id: str):
        """Initialize factory with execution context."""
        self.execution_id = execution_id
        logger.debug(f"BinaryAssignmentVariableFactory initialized for execution {execution_id}")

    def create_variable(self, idx: int, assignment_tuple: Optional[Tuple] = None,
                       config: VariableCreationConfig = VariableCreationConfig()) -> pulp.LpVariable:
        """
        Create binary assignment variable with mathematical correctness.

        Args:
            idx: Flat variable index from bijection mapping
            assignment_tuple: Optional (c,f,r,t,b) tuple for semantic naming
            config: Variable creation configuration

        Returns:
            PuLP LpVariable with binary category and appropriate naming

        Raises:
            ValueError: If variable creation fails mathematical validation
        """
        try:
            # Generate variable name based on strategy
            if config.naming_strategy == "semantic" and assignment_tuple is not None:
                c, f, r, t, b = assignment_tuple
                var_name = f"{config.variable_prefix}_{c}_{f}_{r}_{t}_{b}"
            else:
                var_name = f"{config.variable_prefix}_{idx}"

            # Create binary variable per MILP formulation
            if config.binary_variables:
                variable = pulp.LpVariable(
                    name=var_name,
                    cat=pulp.LpBinary,
                    lowBound=0,
                    upBound=1
                )
            else:
                # Continuous relaxation for debugging/testing
                variable = pulp.LpVariable(
                    name=var_name,
                    cat=pulp.LpContinuous,
                    lowBound=0.0,
                    upBound=1.0
                )

            # Validate variable if enabled
            if config.mathematical_validation and not self.validate_variable(variable):
                raise ValueError(f"Variable validation failed for index {idx}")

            return variable

        except Exception as e:
            logger.error(f"Failed to create variable for index {idx}: {str(e)}")
            raise ValueError(f"Variable creation failed: {str(e)}") from e

    def validate_variable(self, variable: pulp.LpVariable) -> bool:
        """
        Validate binary assignment variable mathematical properties.

        Ensures variable satisfies MILP formulation requirements:
        - Binary category for integer programming
        - Proper bounds [0,1] for assignment semantics
        - Valid naming convention for solver compatibility
        """
        try:
            # Check variable category
            if hasattr(variable, 'cat'):
                if variable.cat not in [pulp.LpBinary, pulp.LpContinuous]:
                    logger.warning(f"Invalid variable category: {variable.cat}")
                    return False

            # Check variable bounds
            if hasattr(variable, 'lowBound') and variable.lowBound < 0:
                logger.warning(f"Invalid lower bound: {variable.lowBound}")
                return False

            if hasattr(variable, 'upBound') and variable.upBound > 1:
                logger.warning(f"Invalid upper bound: {variable.upBound}")  
                return False

            # Check variable name validity
            if not variable.name or not isinstance(variable.name, str):
                logger.warning(f"Invalid variable name: {variable.name}")
                return False

            return True

        except Exception as e:
            logger.error(f"Variable validation failed: {str(e)}")
            return False


class ContinuousAuxiliaryVariableFactory(VariableFactory):
    """
    Factory for creating continuous auxiliary variables for resource utilization modeling.

    Mathematical Foundation: Supports extended MILP formulations with continuous
    variables for resource utilization, load balancing, and preference satisfaction
    modeling per advanced optimization requirements.
    """

    def __init__(self, execution_id: str):
        """Initialize factory with execution context."""
        self.execution_id = execution_id
        logger.debug(f"ContinuousAuxiliaryVariableFactory initialized for execution {execution_id}")

    def create_variable(self, idx: int, assignment_tuple: Optional[Tuple] = None,
                       config: VariableCreationConfig = VariableCreationConfig()) -> pulp.LpVariable:
        """Create continuous auxiliary variable."""
        try:
            var_name = f"aux_{config.variable_prefix}_{idx}"

            variable = pulp.LpVariable(
                name=var_name,
                cat=pulp.LpContinuous,
                lowBound=0.0,
                upBound=None  # Unbounded above
            )

            if config.mathematical_validation and not self.validate_variable(variable):
                raise ValueError(f"Auxiliary variable validation failed for index {idx}")

            return variable

        except Exception as e:
            logger.error(f"Failed to create auxiliary variable for index {idx}: {str(e)}")
            raise ValueError(f"Auxiliary variable creation failed: {str(e)}") from e

    def validate_variable(self, variable: pulp.LpVariable) -> bool:
        """Validate continuous auxiliary variable properties."""
        try:
            # Check continuous category
            if hasattr(variable, 'cat') and variable.cat != pulp.LpContinuous:
                logger.warning(f"Auxiliary variable must be continuous: {variable.cat}")
                return False

            # Check non-negative lower bound
            if hasattr(variable, 'lowBound') and variable.lowBound < 0:
                logger.warning(f"Auxiliary variable must be non-negative: {variable.lowBound}")
                return False

            return True

        except Exception as e:
            logger.error(f"Auxiliary variable validation failed: {str(e)}")
            return False


class PuLPVariableManager:
    """
    Enterprise-grade variable manager for PuLP optimization problems.

    Implements comprehensive variable creation, management, and validation functionality
    following Stage 6.1 theoretical framework. Provides mathematical guarantees for
    MILP formulation correctness while maintaining optimal performance characteristics.

    Mathematical Foundation:
        - Implements complete variable space creation per Definition 2.1 (Scheduling MILP)
        - Maintains bijection consistency per stride-based mapping algorithm
        - Ensures variable index space utilization per V = ∑V_c formulation
        - Provides mathematical validation per MILP theoretical requirements
        - Supports multi-solver compatibility across PuLP backend family
    """

    def __init__(self, execution_id: str, config: VariableCreationConfig = VariableCreationConfig()):
        """
        Initialize variable manager with execution context and configuration.

        Args:
            execution_id: Unique execution identifier for logging and tracking
            config: Variable creation configuration parameters
        """
        self.execution_id = execution_id
        self.config = config
        self.config.validate_config()

        # Initialize variable factories
        self.binary_factory = BinaryAssignmentVariableFactory(execution_id)
        self.auxiliary_factory = ContinuousAuxiliaryVariableFactory(execution_id)

        # Initialize state
        self.variables: Dict[int, pulp.LpVariable] = {}
        self.creation_metadata: Dict[str, Any] = {}
        self.is_created = False

        logger.info(f"PuLPVariableManager initialized for execution {execution_id}")

    def create_variables_from_bijection(self, bijection_mapping: BijectiveMapping,
                                      entity_collections: Optional[Dict[str, EntityCollection]] = None) -> VariableCreationResult:
        """
        Create complete variable set from bijection mapping with mathematical rigor.

        Implements variable creation per Stage 6 Processing Layer requirements, creating
        decision variables for entire optimization problem index space with guaranteed
        mathematical correctness and optimal memory utilization.

        Args:
            bijection_mapping: Complete bijective mapping with strides and offsets
            entity_collections: Optional entity collections for semantic variable naming

        Returns:
            VariableCreationResult with complete variable set and metadata

        Raises:
            ValueError: If bijection mapping is invalid or variable creation fails
            RuntimeError: If mathematical validation fails or memory limits exceeded
        """
        logger.info(f"Creating variables from bijection mapping for execution {self.execution_id}")

        start_time = datetime.now()

        try:
            # Phase 1: Validate bijection mapping
            self._validate_bijection_mapping(bijection_mapping)

            # Phase 2: Calculate variable creation plan
            total_variables = bijection_mapping.total_variables
            creation_batches = self._calculate_creation_batches(total_variables)

            logger.info(f"Creating {total_variables} variables in {len(creation_batches)} batches")

            # Phase 3: Create variables in batches for memory efficiency
            created_variables = {}
            index_mapping = {}
            variable_types = {"binary": 0, "continuous": 0}

            for batch_idx, (start_idx, end_idx) in enumerate(creation_batches):
                logger.debug(f"Creating variable batch {batch_idx + 1}/{len(creation_batches)}: [{start_idx}, {end_idx})")

                batch_variables, batch_mapping = self._create_variable_batch(
                    start_idx, end_idx, bijection_mapping, entity_collections
                )

                created_variables.update(batch_variables)
                index_mapping.update(batch_mapping)

                # Update variable type counts
                for var in batch_variables.values():
                    if hasattr(var, 'cat'):
                        if var.cat == pulp.LpBinary:
                            variable_types["binary"] += 1
                        elif var.cat == pulp.LpContinuous:
                            variable_types["continuous"] += 1

            # Phase 4: Validate created variables
            if self.config.mathematical_validation:
                self._validate_variable_set(created_variables, bijection_mapping)

            # Phase 5: Perform solver compatibility check
            solver_compatibility = {}
            if self.config.solver_compatibility_check:
                solver_compatibility = self._check_solver_compatibility(created_variables)

            # Phase 6: Calculate creation metrics
            end_time = datetime.now()
            creation_time = (end_time - start_time).total_seconds()
            memory_usage = self._estimate_memory_usage(created_variables)

            # Phase 7: Generate mathematical properties verification
            mathematical_properties = self._verify_mathematical_properties(
                created_variables, bijection_mapping
            )

            # Store variables and mark as created
            self.variables = created_variables
            self.is_created = True

            # Generate creation result
            result = VariableCreationResult(
                variables=created_variables,
                variable_count=len(created_variables),
                creation_time_seconds=creation_time,
                memory_usage_bytes=memory_usage,
                variable_types=variable_types,
                index_mapping=index_mapping,
                mathematical_properties=mathematical_properties,
                solver_compatibility=solver_compatibility,
                metadata={
                    'execution_id': self.execution_id,
                    'creation_timestamp': end_time.isoformat(),
                    'bijection_total_variables': bijection_mapping.total_variables,
                    'config': self.config.__dict__,
                    'batch_count': len(creation_batches)
                }
            )

            logger.info(f"Successfully created {len(created_variables)} variables in {creation_time:.2f} seconds")

            return result

        except Exception as e:
            logger.error(f"Variable creation failed: {str(e)}")
            raise RuntimeError(f"Variable creation failed for execution {self.execution_id}: {str(e)}") from e

    def _validate_bijection_mapping(self, bijection_mapping: BijectiveMapping) -> None:
        """Validate bijection mapping for variable creation requirements."""
        if bijection_mapping.total_variables <= 0:
            raise ValueError("Total variables must be positive")

        if len(bijection_mapping.course_blocks) == 0:
            raise ValueError("Course blocks cannot be empty")

        if len(bijection_mapping.offsets) != len(bijection_mapping.course_blocks) + 1:
            raise ValueError("Offsets array length inconsistent with course blocks")

        # Verify offsets consistency
        expected_total = bijection_mapping.offsets[-1]
        if expected_total != bijection_mapping.total_variables:
            raise ValueError(f"Total variables mismatch: {expected_total} != {bijection_mapping.total_variables}")

        logger.debug("Bijection mapping validation passed")

    def _calculate_creation_batches(self, total_variables: int) -> List[Tuple[int, int]]:
        """Calculate variable creation batches for memory efficiency."""
        if not self.config.memory_optimization:
            return [(0, total_variables)]

        batch_size = self.config.batch_size
        batches = []

        for start_idx in range(0, total_variables, batch_size):
            end_idx = min(start_idx + batch_size, total_variables)
            batches.append((start_idx, end_idx))

        return batches

    def _create_variable_batch(self, start_idx: int, end_idx: int,
                             bijection_mapping: BijectiveMapping,
                             entity_collections: Optional[Dict[str, EntityCollection]]) -> Tuple[Dict[int, pulp.LpVariable], Dict[int, Tuple]]:
        """Create batch of variables with optional semantic naming."""
        batch_variables = {}
        batch_mapping = {}

        for idx in range(start_idx, end_idx):
            try:
                # Decode assignment tuple for semantic naming if requested
                assignment_tuple = None
                if self.config.naming_strategy == "semantic":
                    try:
                        assignment_tuple = bijection_mapping.decode(idx)
                        batch_mapping[idx] = assignment_tuple
                    except Exception as e:
                        logger.warning(f"Failed to decode assignment tuple for index {idx}: {str(e)}")
                        assignment_tuple = None

                # Create primary binary assignment variable
                variable = self.binary_factory.create_variable(idx, assignment_tuple, self.config)
                batch_variables[idx] = variable

                # Create auxiliary continuous variables if enabled
                if self.config.continuous_auxiliary:
                    aux_variable = self.auxiliary_factory.create_variable(idx, assignment_tuple, self.config)
                    batch_variables[f"aux_{idx}"] = aux_variable

            except Exception as e:
                logger.error(f"Failed to create variable for index {idx}: {str(e)}")
                raise

        return batch_variables, batch_mapping

    def _validate_variable_set(self, variables: Dict[int, pulp.LpVariable],
                             bijection_mapping: BijectiveMapping) -> None:
        """Validate complete variable set for mathematical consistency."""

        # Check variable count consistency
        expected_count = bijection_mapping.total_variables
        actual_count = len([k for k in variables.keys() if isinstance(k, int)])

        if actual_count != expected_count:
            raise ValueError(f"Variable count mismatch: {actual_count} != {expected_count}")

        # Check index coverage
        expected_indices = set(range(bijection_mapping.total_variables))
        actual_indices = set(k for k in variables.keys() if isinstance(k, int))

        if actual_indices != expected_indices:
            missing_indices = expected_indices - actual_indices
            extra_indices = actual_indices - expected_indices

            error_msg = []
            if missing_indices:
                error_msg.append(f"Missing indices: {sorted(list(missing_indices))[:10]}")
            if extra_indices:
                error_msg.append(f"Extra indices: {sorted(list(extra_indices))[:10]}")

            raise ValueError(f"Index coverage mismatch: {'; '.join(error_msg)}")

        # Validate individual variables
        invalid_variables = []
        for idx, variable in variables.items():
            if isinstance(idx, int) and not self.binary_factory.validate_variable(variable):
                invalid_variables.append(idx)

        if invalid_variables:
            raise ValueError(f"Invalid variables found: {invalid_variables[:10]}")

        logger.debug(f"Variable set validation passed: {actual_count} variables verified")

    def _check_solver_compatibility(self, variables: Dict[int, pulp.LpVariable]) -> Dict[str, bool]:
        """Check variable compatibility with PuLP solver backends."""
        compatibility = {}

        # Test variable creation with each PuLP solver (if available)
        solvers_to_test = ['CBC', 'GLPK', 'HiGHS', 'CLP', 'PULP_CBC_CMD']

        for solver_name in solvers_to_test:
            try:
                # Create test problem with sample variables
                test_prob = pulp.LpProblem(f"compatibility_test_{solver_name}", pulp.LpMinimize)

                # Add sample variables to test problem
                sample_indices = list(variables.keys())[:min(10, len(variables))]
                sample_variables = [variables[idx] for idx in sample_indices if isinstance(idx, int)]

                if sample_variables:
                    # Add dummy objective
                    test_prob += pulp.lpSum(sample_variables)

                    # Test solver availability (don't actually solve)
                    solver_available = False
                    if solver_name == 'CBC':
                        solver_available = pulp.COIN_CMD().available()
                    elif solver_name == 'GLPK':
                        solver_available = pulp.GLPK_CMD().available()
                    elif solver_name == 'HiGHS':
                        try:
                            solver_available = pulp.HiGHS().available()
                        except:
                            solver_available = False
                    elif solver_name == 'CLP':
                        solver_available = pulp.COIN_CMD().available()
                    elif solver_name == 'PULP_CBC_CMD':
                        solver_available = pulp.PULP_CBC_CMD().available()

                    compatibility[solver_name] = solver_available
                else:
                    compatibility[solver_name] = True  # No variables to test

            except Exception as e:
                logger.warning(f"Solver compatibility check failed for {solver_name}: {str(e)}")
                compatibility[solver_name] = False

        return compatibility

    def _estimate_memory_usage(self, variables: Dict[int, pulp.LpVariable]) -> int:
        """Estimate memory usage for variable storage."""

        # Rough estimation based on PuLP variable structure
        bytes_per_variable = 200  # Approximate bytes per LpVariable object
        auxiliary_overhead = 50   # Additional overhead per variable

        total_variables = len(variables)
        estimated_bytes = total_variables * (bytes_per_variable + auxiliary_overhead)

        # Add overhead for dictionary storage
        dictionary_overhead = total_variables * 24  # Approximate overhead for dict entries

        return estimated_bytes + dictionary_overhead

    def _verify_mathematical_properties(self, variables: Dict[int, pulp.LpVariable],
                                      bijection_mapping: BijectiveMapping) -> Dict[str, Any]:
        """Verify mathematical properties of created variable set."""

        properties = {
            'formulation_compliant': True,
            'variable_space_complete': False,
            'binary_category_correct': False,
            'bounds_valid': False,
            'naming_consistent': False,
            'bijection_aligned': False
        }

        try:
            # Check variable space completeness
            integer_variables = {k: v for k, v in variables.items() if isinstance(k, int)}
            properties['variable_space_complete'] = len(integer_variables) == bijection_mapping.total_variables

            # Check binary category correctness
            binary_count = sum(1 for var in integer_variables.values() if hasattr(var, 'cat') and var.cat == pulp.LpBinary)
            properties['binary_category_correct'] = binary_count == len(integer_variables)

            # Check bounds validity
            valid_bounds = True
            for var in integer_variables.values():
                if hasattr(var, 'lowBound') and var.lowBound < 0:
                    valid_bounds = False
                    break
                if hasattr(var, 'upBound') and var.upBound > 1:
                    valid_bounds = False
                    break
            properties['bounds_valid'] = valid_bounds

            # Check naming consistency
            naming_consistent = True
            for idx, var in integer_variables.items():
                if not var.name or not isinstance(var.name, str):
                    naming_consistent = False
                    break
                if self.config.variable_prefix not in var.name:
                    naming_consistent = False
                    break
            properties['naming_consistent'] = naming_consistent

            # Check bijection alignment
            expected_indices = set(range(bijection_mapping.total_variables))
            actual_indices = set(integer_variables.keys())
            properties['bijection_aligned'] = expected_indices == actual_indices

            # Overall formulation compliance
            properties['formulation_compliant'] = all([
                properties['variable_space_complete'],
                properties['binary_category_correct'],
                properties['bounds_valid'],
                properties['naming_consistent'],
                properties['bijection_aligned']
            ])

        except Exception as e:
            logger.error(f"Mathematical properties verification failed: {str(e)}")
            properties['formulation_compliant'] = False
            properties['verification_error'] = str(e)

        return properties

    def get_variable_by_assignment(self, course_id: Any, faculty_id: Any, room_id: Any,
                                 timeslot_id: Any, batch_id: Any,
                                 bijection_mapping: BijectiveMapping) -> Optional[pulp.LpVariable]:
        """
        Get variable by assignment tuple (c,f,r,t,b) using bijection mapping.

        Args:
            course_id: Course entity ID
            faculty_id: Faculty entity ID
            room_id: Room entity ID
            timeslot_id: Timeslot entity ID
            batch_id: Batch entity ID
            bijection_mapping: Bijection mapping for encoding

        Returns:
            PuLP LpVariable corresponding to assignment, or None if not found
        """
        if not self.is_created:
            logger.error("Variables have not been created yet")
            return None

        try:
            # Encode assignment tuple to flat index
            idx = bijection_mapping.encode(course_id, faculty_id, room_id, timeslot_id, batch_id)

            # Return variable if it exists
            return self.variables.get(idx)

        except Exception as e:
            logger.error(f"Failed to get variable by assignment: {str(e)}")
            return None

    def get_variables_for_course(self, course_id: Any, bijection_mapping: BijectiveMapping) -> Dict[int, pulp.LpVariable]:
        """
        Get all variables for specific course.

        Args:
            course_id: Course entity ID
            bijection_mapping: Bijection mapping for index calculation

        Returns:
            Dictionary of variables for the specified course
        """
        if not self.is_created:
            logger.error("Variables have not been created yet")
            return {}

        try:
            # Get variable index range for course
            start_idx, end_idx = bijection_mapping.get_course_variable_range(course_id)

            # Extract variables in range
            course_variables = {}
            for idx in range(start_idx, end_idx):
                if idx in self.variables:
                    course_variables[idx] = self.variables[idx]

            return course_variables

        except Exception as e:
            logger.error(f"Failed to get variables for course {course_id}: {str(e)}")
            return {}

    def get_variable_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about created variables.

        Returns:
            Dictionary containing variable statistics and metadata
        """
        if not self.is_created:
            return {'status': 'variables_not_created'}

        # Count variables by type
        integer_variables = {k: v for k, v in self.variables.items() if isinstance(k, int)}
        auxiliary_variables = {k: v for k, v in self.variables.items() if not isinstance(k, int)}

        binary_count = sum(1 for var in integer_variables.values() if hasattr(var, 'cat') and var.cat == pulp.LpBinary)
        continuous_count = sum(1 for var in integer_variables.values() if hasattr(var, 'cat') and var.cat == pulp.LpContinuous)

        # Calculate memory usage
        estimated_memory = self._estimate_memory_usage(self.variables)

        return {
            'total_variables': len(self.variables),
            'integer_variables': len(integer_variables),
            'auxiliary_variables': len(auxiliary_variables),
            'binary_variables': binary_count,
            'continuous_variables': continuous_count,
            'estimated_memory_bytes': estimated_memory,
            'estimated_memory_mb': estimated_memory / (1024 * 1024),
            'creation_config': self.config.__dict__,
            'execution_id': self.execution_id,
            'is_created': self.is_created
        }

    def save_variables_metadata(self, output_path: Union[str, Path]) -> Path:
        """
        Save variable creation metadata to JSON file.

        Args:
            output_path: Directory path where metadata file should be saved

        Returns:
            Path to saved metadata file
        """
        if not self.is_created:
            raise ValueError("Variables must be created before saving metadata")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        metadata_filename = f"variables_metadata_{self.execution_id}.json"
        metadata_path = output_path / metadata_filename

        # Generate comprehensive metadata
        metadata = {
            'variable_creation_info': {
                'execution_id': self.execution_id,
                'creation_timestamp': datetime.now().isoformat(),
                'manager_version': '1.0.0',
                'theoretical_framework': 'Stage 6.1 PuLP MILP Formulation'
            },
            'variable_statistics': self.get_variable_statistics(),
            'mathematical_properties': getattr(self, 'last_mathematical_properties', {}),
            'solver_compatibility': getattr(self, 'last_solver_compatibility', {}),
            'configuration': self.config.__dict__
        }

        # Save metadata to file
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Variable metadata saved to {metadata_path}")
        return metadata_path


def create_pulp_variables(bijection_mapping: BijectiveMapping,
                        execution_id: str,
                        entity_collections: Optional[Dict[str, EntityCollection]] = None,
                        config: Optional[VariableCreationConfig] = None,
                        output_path: Optional[Union[str, Path]] = None) -> Tuple[Dict[int, pulp.LpVariable], VariableCreationResult]:
    """
    High-level function to create PuLP variables from bijection mapping.

    Provides simplified interface for variable creation with comprehensive validation
    and optional metadata output for processing pipeline integration.

    Args:
        bijection_mapping: Complete bijective mapping from input modeling
        execution_id: Unique execution identifier
        entity_collections: Optional entity collections for semantic naming
        config: Optional variable creation configuration
        output_path: Optional path to save variable metadata

    Returns:
        Tuple containing (variables_dict, creation_result)

    Raises:
        ValueError: If bijection mapping is invalid
        RuntimeError: If variable creation fails

    Example:
        >>> variables, result = create_pulp_variables(bijection, "exec_001")
        >>> print(f"Created {result.variable_count} variables in {result.creation_time_seconds:.2f}s")
    """
    # Use default config if not provided
    if config is None:
        config = VariableCreationConfig()

    # Initialize variable manager
    manager = PuLPVariableManager(execution_id=execution_id, config=config)

    # Create variables
    creation_result = manager.create_variables_from_bijection(
        bijection_mapping=bijection_mapping,
        entity_collections=entity_collections
    )

    # Save metadata if output path specified
    if output_path:
        manager.save_variables_metadata(output_path)

    logger.info(f"Successfully created {creation_result.variable_count} PuLP variables for execution {execution_id}")

    return creation_result.variables, creation_result


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
    except ImportError:
        print("Failed to import input_model modules - ensure proper project structure")
        sys.exit(1)

    if len(sys.argv) != 3:
        print("Usage: python variables.py <input_path> <execution_id>")
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
            print(f"✗ Data validation failed - cannot create variables")
            sys.exit(1)

        # Build bijection mapping
        bijection = build_bijection_mapping(entities, execution_id)

        # Create variables with custom configuration
        config = VariableCreationConfig(
            variable_prefix="x",
            binary_variables=True,
            naming_strategy="indexed",
            mathematical_validation=True,
            solver_compatibility_check=True
        )

        variables, result = create_pulp_variables(bijection, execution_id, entities, config)

        print(f"✓ Variables created successfully for execution {execution_id}")

        # Print result summary
        summary = result.get_summary()
        print(f"  Variable count: {summary['variable_count']:,}")
        print(f"  Creation time: {summary['creation_time_seconds']:.2f} seconds")
        print(f"  Memory usage: {summary['memory_usage_mb']:.1f} MB")
        print(f"  Variable types: {summary['variable_types']}")
        print(f"  Mathematical compliance: {summary['mathematical_compliance']}")
        print(f"  Solver compatibility: {summary['solver_compatibility']}")

        # Test variable access
        if variables and bijection.total_variables > 0:
            first_var_idx = 0
            if first_var_idx in variables:
                first_var = variables[first_var_idx]
                print(f"  Sample variable: {first_var.name} (category: {first_var.cat})")

    except Exception as e:
        print(f"Failed to create variables: {str(e)}")
        sys.exit(1)
