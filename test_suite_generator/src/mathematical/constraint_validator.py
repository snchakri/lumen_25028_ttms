"""
Constraint Validation Engine

Validates generated data against foundation constraints using
symbolic mathematics and runtime validation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from enum import Enum
import logging

from .symbolic_math import SymbolicMath, SymbolicConstraint, ConstraintType

logger = logging.getLogger(__name__)


class ValidationMode(Enum):
    """Validation execution modes."""
    PRE_GENERATION = "pre_generation"
    PER_ENTITY = "per_entity"
    POST_GENERATION = "post_generation"
    PROOF_GENERATION = "proof_generation"


@dataclass
class ValidationResult:
    """
    Result of constraint validation.
    
    Attributes:
        constraint_id: Constraint identifier
        passed: Whether constraint was satisfied
        values: Values used in validation
        message: Human-readable message
        counter_example: Counter-example if violated
    """
    constraint_id: str
    passed: bool
    values: Dict[str, Any]
    message: str
    counter_example: Optional[Dict[str, Any]] = None


@dataclass
class ValidationReport:
    """
    Comprehensive validation report.
    
    Attributes:
        mode: Validation mode used
        total_constraints: Total constraints checked
        passed: Number of constraints passed
        failed: Number of constraints failed
        results: Individual validation results
        violations: List of violated constraints
    """
    mode: ValidationMode
    total_constraints: int
    passed: int
    failed: int
    results: List[ValidationResult]
    violations: List[SymbolicConstraint]
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate percentage."""
        if self.total_constraints == 0:
            return 0.0
        return (self.passed / self.total_constraints) * 100.0
    
    @property
    def is_valid(self) -> bool:
        """Check if all constraints passed."""
        return self.failed == 0


class ConstraintValidator:
    """
    Constraint validation engine.
    
    Validates generated data against symbolic constraints from
    foundation documents. Supports multiple validation modes and
    detailed reporting.
    """
    
    def __init__(self, symbolic_math: Optional[SymbolicMath] = None):
        """
        Initialize constraint validator.
        
        Args:
            symbolic_math: SymbolicMath engine (creates new if None)
        """
        self.symbolic_math = symbolic_math or SymbolicMath()
        self._validation_cache: Dict[str, ValidationResult] = {}
        self._enabled_modes: Set[ValidationMode] = {
            ValidationMode.PRE_GENERATION,
            ValidationMode.PER_ENTITY,
            ValidationMode.POST_GENERATION
        }
        logger.info("ConstraintValidator initialized")
    
    def enable_mode(self, mode: ValidationMode) -> None:
        """Enable a validation mode."""
        self._enabled_modes.add(mode)
        logger.debug(f"Enabled validation mode: {mode.value}")
    
    def disable_mode(self, mode: ValidationMode) -> None:
        """Disable a validation mode."""
        self._enabled_modes.discard(mode)
        logger.debug(f"Disabled validation mode: {mode.value}")
    
    def is_mode_enabled(self, mode: ValidationMode) -> bool:
        """Check if a validation mode is enabled."""
        return mode in self._enabled_modes
    
    def validate_entity(
        self,
        entity_data: Dict[str, Any],
        entity_type: str,
        mode: ValidationMode = ValidationMode.PER_ENTITY
    ) -> ValidationReport:
        """
        Validate a single entity against applicable constraints.
        
        Args:
            entity_data: Entity field values
            entity_type: Type of entity (e.g., 'student', 'course')
            mode: Validation mode
            
        Returns:
            ValidationReport with results
            
        Example:
            >>> validator = ConstraintValidator()
            >>> report = validator.validate_entity(
            ...     {'credit_hours': 3, 'course_code': 'CS101'},
            ...     'course'
            ... )
            >>> print(f"Valid: {report.is_valid}")
        """
        if not self.is_mode_enabled(mode):
            logger.debug(f"Validation mode {mode.value} is disabled, skipping")
            return ValidationReport(
                mode=mode,
                total_constraints=0,
                passed=0,
                failed=0,
                results=[],
                violations=[]
            )
        
        # Get all constraints
        results: List[ValidationResult] = []
        violations: List[SymbolicConstraint] = []
        
        for constraint_id in self.symbolic_math.list_constraints():
            constraint = self.symbolic_math.get_constraint(constraint_id)
            if constraint is None:
                continue
            
            # Check if all required variables are present
            missing_vars = set(constraint.variables) - set(entity_data.keys())
            if missing_vars:
                # Skip if variables not applicable to this entity
                logger.debug(
                    f"Skipping {constraint_id}: missing variables {missing_vars}"
                )
                continue
            
            # Evaluate constraint
            try:
                passed = self.symbolic_math.evaluate_constraint(
                    constraint_id,
                    entity_data
                )
                
                result = ValidationResult(
                    constraint_id=constraint_id,
                    passed=passed,
                    values=entity_data.copy(),
                    message=f"{'PASS' if passed else 'FAIL'}: {constraint.name}",
                    counter_example=entity_data if not passed else None
                )
                results.append(result)
                
                if not passed:
                    violations.append(constraint)
                    
            except Exception as e:
                logger.error(
                    f"Error evaluating constraint {constraint_id}: {e}"
                )
                result = ValidationResult(
                    constraint_id=constraint_id,
                    passed=False,
                    values=entity_data.copy(),
                    message=f"ERROR: {str(e)}",
                    counter_example=entity_data
                )
                results.append(result)
                violations.append(constraint)
        
        report = ValidationReport(
            mode=mode,
            total_constraints=len(results),
            passed=sum(1 for r in results if r.passed),
            failed=sum(1 for r in results if not r.passed),
            results=results,
            violations=violations
        )
        
        if not report.is_valid:
            logger.warning(
                f"Entity validation failed: {report.failed}/{report.total_constraints} "
                f"constraints violated"
            )
        
        return report
    
    def validate_batch(
        self,
        entities: List[Dict[str, Any]],
        entity_type: str,
        mode: ValidationMode = ValidationMode.POST_GENERATION
    ) -> ValidationReport:
        """
        Validate multiple entities.
        
        Args:
            entities: List of entity data dictionaries
            entity_type: Type of entities
            mode: Validation mode
            
        Returns:
            Aggregated ValidationReport
        """
        if not self.is_mode_enabled(mode):
            logger.debug(f"Validation mode {mode.value} is disabled, skipping")
            return ValidationReport(
                mode=mode,
                total_constraints=0,
                passed=0,
                failed=0,
                results=[],
                violations=[]
            )
        
        all_results: List[ValidationResult] = []
        all_violations: List[SymbolicConstraint] = []
        
        for i, entity in enumerate(entities):
            report = self.validate_entity(entity, entity_type, mode)
            all_results.extend(report.results)
            all_violations.extend(report.violations)
            
            if not report.is_valid:
                logger.warning(
                    f"Entity {i} failed validation: {report.failed} violations"
                )
        
        # Remove duplicate violations
        unique_violations: List[SymbolicConstraint] = []
        seen_ids: Set[str] = set()
        for v in all_violations:
            if v.constraint_id not in seen_ids:
                unique_violations.append(v)
                seen_ids.add(v.constraint_id)
        
        report = ValidationReport(
            mode=mode,
            total_constraints=len(all_results),
            passed=sum(1 for r in all_results if r.passed),
            failed=sum(1 for r in all_results if not r.passed),
            results=all_results,
            violations=unique_violations
        )
        
        logger.info(
            f"Batch validation complete: {report.passed}/{report.total_constraints} "
            f"passed ({report.pass_rate:.1f}%)"
        )
        
        return report
    
    def validate_configuration(
        self,
        config: Dict[str, Any]
    ) -> ValidationReport:
        """
        Validate configuration values before generation.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            ValidationReport
        """
        return self.validate_entity(
            config,
            "configuration",
            ValidationMode.PRE_GENERATION
        )
    
    def validate_hard_constraints_only(
        self,
        entity_data: Dict[str, Any],
        entity_type: str
    ) -> ValidationReport:
        """
        Validate only hard constraints (must be satisfied).
        
        Args:
            entity_data: Entity field values
            entity_type: Type of entity
            
        Returns:
            ValidationReport with only hard constraints
        """
        # Temporarily filter to hard constraints only
        hard_constraint_ids = self.symbolic_math.list_constraints(
            ConstraintType.HARD
        )
        
        results = []
        violations = []
        
        for constraint_id in hard_constraint_ids:
            constraint = self.symbolic_math.get_constraint(constraint_id)
            if constraint is None:
                continue
            
            # Check if all required variables are present
            missing_vars = set(constraint.variables) - set(entity_data.keys())
            if missing_vars:
                continue
            
            # Evaluate constraint
            try:
                passed = self.symbolic_math.evaluate_constraint(
                    constraint_id,
                    entity_data
                )
                
                result = ValidationResult(
                    constraint_id=constraint_id,
                    passed=passed,
                    values=entity_data.copy(),
                    message=f"{'PASS' if passed else 'FAIL'}: {constraint.name}",
                    counter_example=entity_data if not passed else None
                )
                results.append(result)
                
                if not passed:
                    violations.append(constraint)
                    
            except Exception as e:
                logger.error(
                    f"Error evaluating constraint {constraint_id}: {e}"
                )
                result = ValidationResult(
                    constraint_id=constraint_id,
                    passed=False,
                    values=entity_data.copy(),
                    message=f"ERROR: {str(e)}",
                    counter_example=entity_data
                )
                results.append(result)
                violations.append(constraint)
        
        report = ValidationReport(
            mode=ValidationMode.PER_ENTITY,
            total_constraints=len(results),
            passed=sum(1 for r in results if r.passed),
            failed=sum(1 for r in results if not r.passed),
            results=results,
            violations=violations
        )
        
        return report
    
    def clear_cache(self) -> None:
        """Clear validation result cache."""
        self._validation_cache.clear()
        logger.debug("Cleared validation cache")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get validation statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_constraints": len(self.symbolic_math.constraints),
            "hard_constraints": len(
                self.symbolic_math.list_constraints(ConstraintType.HARD)
            ),
            "soft_constraints": len(
                self.symbolic_math.list_constraints(ConstraintType.SOFT)
            ),
            "preference_constraints": len(
                self.symbolic_math.list_constraints(ConstraintType.PREFERENCE)
            ),
            "enabled_modes": [mode.value for mode in self._enabled_modes],
            "cache_size": len(self._validation_cache)
        }
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"ConstraintValidator("
            f"constraints={stats['total_constraints']}, "
            f"hard={stats['hard_constraints']}, "
            f"soft={stats['soft_constraints']})"
        )
