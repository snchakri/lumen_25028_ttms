"""
Core data models for Stage-1 Input Validation.

Implements formal data structures from theoretical foundations:
- Schema definitions from hei_timetabling_datamodel.sql
- Validation types per Definition 9.4
- Mathematical types for theorem verification
- Reference graph structures for integrity analysis
"""

from .schema_definitions import MANDATORY_FILES, OPTIONAL_FILES, TABLE_SCHEMAS, ENUM_DEFINITIONS
from .validation_types import ValidationStatus, ErrorSeverity, ValidationResult, ErrorReport
from .mathematical_types import ComplexityBounds, QualityVector, TheoremVerification
from .reference_graph import ReferenceGraph

__all__ = [
    "MANDATORY_FILES",
    "OPTIONAL_FILES",
    "TABLE_SCHEMAS",
    "ENUM_DEFINITIONS",
    "ValidationStatus",
    "ErrorSeverity",
    "ValidationResult",
    "ErrorReport",
    "ComplexityBounds",
    "QualityVector",
    "TheoremVerification",
    "ReferenceGraph",
]





