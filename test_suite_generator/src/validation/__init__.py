"""
Validation Framework - Seven-Layer Validation System

This package implements the comprehensive seven-layer validation framework
as specified in DESIGN_PART_4_VALIDATION_FRAMEWORK.md.

Validation Layers:
    L1: Structural Validation (UUID, NOT NULL, data types, foreign keys)
    L2: Domain Validation (CHECK constraints, enums, patterns, ranges)
    L3: Temporal Validation (date/time formats, ordering, overlaps)
    L4: Relational Validation (referential integrity, cardinality)
    L5: Business Validation (credit limits, workload, prerequisites)
    L6: LTREE Validation (hierarchy paths, consistency, depth)
    L7: Scheduling Feasibility (resource adequacy, constraint satisfaction)

Architecture:
    - Base validator classes with common validation logic
    - Individual layer validators implementing specific checks
    - Validation orchestrator for sequential execution
    - Error classification and recovery strategies
    - Comprehensive validation reporting

Compliance:
    - 100% violation detection rate required
    - All foundation constraints validated
    - All schema constraints validated
    - Mathematical rigor in all validations
"""

from src.validation.error_models import (
    ValidationError,
    ValidationResult,
    ErrorSeverity,
    ErrorCategory,
)
from src.validation.base_validator import BaseValidator
from src.validation.validation_context import ValidationContext

__all__ = [
    "ValidationError",
    "ValidationResult",
    "ErrorSeverity",
    "ErrorCategory",
    "BaseValidator",
    "ValidationContext",
]

__version__ = "1.0.0"
__author__ = "Test Data Generator Team"
