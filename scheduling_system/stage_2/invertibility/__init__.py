"""
Invertibility Module for Stage-2 Batching System
Implements perfect reconstruction and bijective transformation verification
"""

from stage_2.invertibility.audit_trail import TransformationAuditor
from stage_2.invertibility.entropy_validation import compute_information_entropy
from stage_2.invertibility.canonical_ordering import (
    get_canonical_student_order,
    resolve_assignment_ties
)
from stage_2.invertibility.reconstruction import (
    verify_transformation_bijectivity,
    reconstruct_input_state
)

__all__ = [
    'TransformationAuditor',
    'compute_information_entropy',
    'get_canonical_student_order',
    'resolve_assignment_ties',
    'verify_transformation_bijectivity',
    'reconstruct_input_state'
]

